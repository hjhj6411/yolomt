#!/usr/bin/env python3
"""
YOLOMT TensorRT Inference Script
Optimized for Jetson Xavier NX / Orin Nano
Fixed for SSH/Headless environments
"""

import argparse
import time
import cv2
import numpy as np
import sys
import math
import os

# TensorRT & PyCUDA
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    print("✅ TensorRT & PyCUDA imported successfully")
except ImportError:
    print("❌ ERROR: TensorRT or PyCUDA not found!")
    print("   Run: pip install pycuda")
    sys.exit(1)

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print(f"🔄 Loading Engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        # 메모리 할당
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        
        print(f"✅ Engine Loaded! Inputs: {len(self.inputs)}, Outputs: {len(self.outputs)}")

    def infer(self, img_in):
        # 1. 입력 데이터 호스트로 복사
        np.copyto(self.inputs[0]['host'], img_in.ravel())
        
        # 2. Host -> Device (Async)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. 추론 실행 (V3 or V2)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 4. Device -> Host (Async)
        results = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            results.append(out['host'])
        
        # 5. 동기화 (대기)
        self.stream.synchronize()
        return results

class YOLOMTDetector:
    def __init__(self, engine_path, conf_thresh=0.25, img_size=320):
        self.trt_net = TensorRTInference(engine_path)
        self.conf_thresh = conf_thresh
        self.img_size = img_size
        self.stride = 16  # YOLOMT의 Stride (입력/출력 비율)

    def preprocess(self, img):
        # Letterbox Resize (비율 유지 리사이즈)
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        nw, nh = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img, (nw, nh))
        
        # Canvas 생성 (회색)
        img_pad = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        dw, dh = (self.img_size - nw) // 2, (self.img_size - nh) // 2
        img_pad[dh:dh+nh, dw:dw+nw] = img_resized
        
        # Normalize & CHW
        img_in = img_pad.astype(np.float32) / 255.0
        img_in = img_in.transpose(2, 0, 1) # HWC -> CHW
        img_in = np.expand_dims(img_in, axis=0) # Batch dim
        
        return np.ascontiguousarray(img_in), scale, (dw, dh)

    def postprocess(self, outputs, scale, pad):
        # outputs[0]: det_out (Flat), outputs[1]: kpt_out (Flat)
        # 중요: ONNX Export 순서에 따라 인덱스가 다를 수 있음 (보통 Det, Kpt 순)
        
        grid_h = self.img_size // self.stride
        grid_w = self.img_size // self.stride
        
        # 🔧 [수정] 1차원 배열을 정확한 차원으로 Reshape
        # Det: (1, 6, H, W), Kpt: (1, 204, H, W)
        det_out = outputs[0].reshape(1, 6, grid_h, grid_w)
        kpt_out = outputs[1].reshape(1, 204, grid_h, grid_w)
        
        boxes, kpts_list, scores = [], [], []
        dw, dh = pad
        
        # Grid 순회 (속도를 위해 Numpy 연산 권장하지만, 가독성을 위해 for문 유지)
        # *실제 고속화 시에는 numpy vectorization 필요
        for r in range(grid_h):
            for c in range(grid_w):
                conf = det_out[0, 4, r, c]
                
                # Sigmoid 적용 (학습 때 BCEWithLogitsLoss 썼으므로)
                conf = 1 / (1 + np.exp(-conf))
                
                if conf < self.conf_thresh:
                    continue
                
                # Box Decoding
                bx = (c + det_out[0, 0, r, c]) * self.stride
                by = (r + det_out[0, 1, r, c]) * self.stride
                bw = det_out[0, 2, r, c] * self.img_size
                bh = det_out[0, 3, r, c] * self.img_size
                
                # 원본 좌표 복원
                x1 = (bx - bw/2 - dw) / scale
                y1 = (by - bh/2 - dh) / scale
                x2 = (bx + bw/2 - dw) / scale
                y2 = (by + bh/2 - dh) / scale
                
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
                
                # Keypoint Decoding
                curr_kpts = []
                k_vec = kpt_out[0, :, r, c] # (204,)
                for k in range(0, 204, 3):
                    kx = (c + k_vec[k]) * self.stride
                    ky = (r + k_vec[k+1]) * self.stride
                    kconf = 1 / (1 + np.exp(-k_vec[k+2]))
                    
                    kx = (kx - dw) / scale
                    ky = (ky - dh) / scale
                    curr_kpts.append((kx, ky, kconf))
                
                kpts_list.append(curr_kpts)
                
        return boxes, kpts_list, scores
    
    def detect(self, frame):
        """전체 감지 파이프라인 (누락된 메서드 추가)"""
        # 1. 전처리
        img_in, scale, pad = self.preprocess(frame)
        
        # 2. 추론
        outputs = self.trt_net.infer(img_in)
        
        # 3. 후처리
        boxes, kpts_list, scores = self.postprocess(outputs, scale, pad)
        
        return boxes, kpts_list, scores

class HeadPoseEstimator:
    """PnP Head Pose"""
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))

    def estimate(self, kpts, img_shape):
        h, w = img_shape[:2]
        if self.camera_matrix is None:
            focal_length = w
            center = (w/2, h/2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], 
                dtype=np.float64
            )
            
        # 6 Keypoints: Nose, Chin, L-Eye, R-Eye, L-Mouth, R-Mouth
        # 인덱스: 30, 8, 36, 45, 48, 54
        indices = [30, 8, 36, 45, 48, 54]
        image_points = []
        for idx in indices:
            if idx < len(kpts):
                image_points.append([kpts[idx][0], kpts[idx][1]])
        
        if len(image_points) < 6:
            return None
            
        image_points = np.array(image_points, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs)
        
        if not success: return None
        
        rmat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
        return euler[0][0], euler[1][0], euler[2][0] # Pitch, Yaw, Roll

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    """GStreamer 파이프라인 (Jetson 카메라용)"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Path to .engine file')
    parser.add_argument('--input', type=str, default='0', help='Camera index or video file')
    parser.add_argument('--output', type=str, default=None, help='Output video file path (optional)')
    parser.add_argument('--headless', action='store_true', help='Run without display (SSH mode)')
    parser.add_argument('--gstreamer', action='store_true', help='Use GStreamer pipeline for Jetson camera')
    parser.add_argument('--img-size', type=int, default=320, help='Model input size')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    
    # 모델 로드
    detector = YOLOMTDetector(args.engine, conf_thresh=args.conf_thresh, img_size=args.img_size)
    pose_est = HeadPoseEstimator()
    
    # 카메라 초기화
    if args.gstreamer:
        # Jetson CSI 카메라용 GStreamer 파이프라인
        src = gstreamer_pipeline(sensor_id=int(args.input) if args.input.isdigit() else 0)
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
    else:
        # 일반 USB 카메라 또는 비디오 파일
        src = int(args.input) if args.input.isdigit() else args.input
        cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {args.input}")
        sys.exit(1)
    
    # 비디오 출력 설정
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"📹 Saving output to: {args.output}")
    
    # 윈도우 생성 (headless가 아닐 때만)
    if not args.headless:
        try:
            cv2.namedWindow("YOLOMT", cv2.WINDOW_NORMAL)
        except:
            print("⚠️  Cannot create window (running in headless mode)")
            args.headless = True

    print("🚀 Running Inference...")
    frame_count = 0
    fps_avg = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            t0 = time.time()
            
            # 1. 추론
            boxes, kpts_list, scores = detector.detect(frame)
            
            # 2. 그리기
            for box, kpts, score in zip(boxes, kpts_list, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 랜드마크
                for kx, ky, kc in kpts:
                    if kc > 0.5:
                        cv2.circle(frame, (int(kx), int(ky)), 2, (0, 0, 255), -1)
                
                # Head Pose
                pose = pose_est.estimate(kpts, frame.shape)
                if pose:
                    pitch, yaw, roll = pose
                    label = f"Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}"
                    cv2.putText(frame, label, (x1, y1-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # FPS 계산
            fps = 1.0 / (time.time() - t0)
            fps_avg = fps_avg * 0.9 + fps * 0.1 if frame_count > 0 else fps
            cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            frame_count += 1
            
            # 3. 출력
            if writer:
                writer.write(frame)
            
            if not args.headless:
                cv2.imshow("YOLOMT", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                # Headless 모드에서는 주기적으로 상태 출력
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: {fps_avg:.1f} FPS, {len(boxes)} faces detected")
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"\n✅ Processed {frame_count} frames (avg {fps_avg:.1f} FPS)")

if __name__ == "__main__":
    main()
