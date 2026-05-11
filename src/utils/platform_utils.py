"""
플랫폼별 최적 설정 유틸리티

OS / 하드웨어를 자동 감지하여 PyTorch device, DataLoader 워커 수,
병렬 프로세스 수 등의 최적값을 반환합니다.

지원 플랫폼:
  - macOS Apple Silicon (arm64) : MPS 가속, spawn 기반 멀티프로세싱
  - Windows x86_64              : CUDA 우선, CPU 폴백
  - Linux x86_64                : CUDA 우선, CPU 폴백
"""

import os
import platform

import torch


# ── Device 감지 ──────────────────────────────────────────────────────────────

def get_device() -> str:
    """
    CUDA → MPS → CPU 우선순위로 사용 가능한 최적 device 문자열을 반환합니다.

    Returns:
        "cuda" | "mps" | "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    _mps = getattr(torch.backends, "mps", None)
    if _mps is not None and _mps.is_available():
        return "mps"
    return "cpu"


def configure_torch(device: str | None = None) -> None:
    """
    device에 맞는 전역 최적화 옵션을 설정합니다.

    - CUDA  : cudnn.benchmark=True (LSTM 등 고정 입력 크기에 유효)
    - MPS   : PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (메모리 한도 해제)
    - CPU   : 스레드 수를 OMP_NUM_THREADS 환경변수에 따라 제한
    """
    if device is None:
        device = get_device()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    elif device == "mps":
        # MPS 메모리 한도를 해제해 OOM 오류 방지
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    # 병렬 subprocess 환경: OMP_NUM_THREADS 가 설정돼 있으면 준수
    _omp = os.environ.get("OMP_NUM_THREADS")
    if _omp:
        try:
            n = int(_omp)
            torch.set_num_threads(n)
            torch.set_num_interop_threads(1)
        except ValueError:
            pass


# ── DataLoader 최적 파라미터 ─────────────────────────────────────────────────

def get_optimal_workers() -> int:
    """
    DataLoader ``num_workers`` 최적값을 반환합니다.

    - CUDA (Windows/Linux GPU): CPU 코어 절반 (최대 8), 비동기 로딩 효과 큼
    - MPS  (macOS Apple Silicon): spawn 방식 오버헤드를 감안해 2로 고정
    - CPU only: 0 (멀티워커 오버헤드가 이득보다 클 수 있음)
    """
    _device = get_device()
    if _device == "cuda":
        return min(8, max(2, (os.cpu_count() or 4) // 2))
    if _device == "mps":
        return 2
    return 0


def get_pin_memory() -> bool:
    """
    DataLoader ``pin_memory`` 최적값을 반환합니다.

    CUDA 환경에서만 ``True``를 반환합니다 (MPS / CPU는 pin_memory 미지원).
    """
    return torch.cuda.is_available()


# ── 병렬 프로세스 수 최적값 ─────────────────────────────────────────────────

def get_optimal_jobs() -> int:
    """
    subprocess / ProcessPoolExecutor 병렬 프로세스 수 최적값을 반환합니다.

    - 총 CPU 코어 수의 절반을 사용 (최소 1, 최대 8)
    - CUDA 환경에서는 GPU 스로틀링 방지를 위해 4로 제한
    """
    cpu = os.cpu_count() or 4
    if torch.cuda.is_available():
        return min(4, max(1, cpu // 2))
    return min(8, max(1, cpu // 2))


# ── 진단 출력 ────────────────────────────────────────────────────────────────

def log_platform_info(logger) -> None:
    """현재 플랫폼 및 선택된 설정을 logger에 INFO 수준으로 출력합니다."""
    _device  = get_device()
    _system  = platform.system()
    _machine = platform.machine()
    _cpu     = os.cpu_count()
    _workers = get_optimal_workers()
    _jobs    = get_optimal_jobs()
    logger.info(
        f"[Platform] OS={_system}/{_machine} | CPUs={_cpu} | "
        f"device={_device} | DataLoader workers={_workers} | parallel jobs={_jobs}"
    )
