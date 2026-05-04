"""
=============================================================================
 암호화폐 딥러닝 트레이딩 봇 - 과거 데이터 수집 및 전처리 파이프라인
=============================================================================
 작성 환경  : Mac M1 Max, conda 가상환경
 의존 패키지: ccxt, pandas, numpy, scikit-learn, TA-Lib
 실행 방법  : python crypto_data_pipeline.py
=============================================================================
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler
import joblib

# ─────────────────────────────────────────────
# 로깅 설정
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. 데이터 수집 클래스
# =============================================================================
class CryptoDataCollector:
    """
    Bybit 거래소에서 OHLCV 데이터를 수집하는 클래스.
    - 페이징(Pagination) 처리로 대량 데이터 수집
    - Rate Limit 준수를 위한 sleep 적용
    - CSV Append 방식으로 메모리 효율적 저장
    - 체크포인트(Checkpoint) 기반 이어받기 수집 지원
    """

    # Bybit는 한 번 요청 시 최대 200개 캔들 반환
    LIMIT_PER_REQUEST: int = 200
    # 요청 간 대기 시간 (초) — Rate Limit 안전 마진
    SLEEP_BETWEEN_REQUESTS: float = 0.5
    # 한 파일에 저장할 캔들 개수 (약 17일치 5분봉 = 4,896개)
    CHUNK_SIZE: int = 5_000

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "5m",
        data_dir: str = "data",
        lookback_years: int = 3,
    ):
        """
        Parameters
        ----------
        symbols        : 수집할 심볼 목록 (예: ["BTC/USDT", "ETH/USDT"])
        timeframe      : 봉 주기 (예: "5m", "1h")
        data_dir       : CSV를 저장할 로컬 디렉토리 경로
        lookback_years : 수집 기간(연도 단위)
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 수집 시작 시각 — 현재 UTC 기준 N년 전
        self.start_dt: datetime = datetime.now(tz=timezone.utc) - timedelta(
            days=365 * lookback_years
        )
        self.start_ts: int = int(self.start_dt.timestamp() * 1000)  # ms 단위

        # ccxt Bybit 거래소 객체 초기화
        self.exchange = ccxt.bybit(
            {
                "enableRateLimit": True,   # ccxt 내장 Rate Limit 활성화
                "options": {"defaultType": "linear"},  # USDT 무기한 선물 마켓
            }
        )
        logger.info(
            "DataCollector 초기화 완료 | 거래소: Bybit | 심볼: %s | 기간: 최근 %d년",
            symbols,
            lookback_years,
        )

    # ─────────────────────────────────────────
    # 내부 유틸리티
    # ─────────────────────────────────────────

    def _get_csv_path(self, symbol: str) -> Path:
        """심볼 이름을 안전한 파일명으로 변환하여 CSV 경로 반환."""
        safe_name = symbol.replace("/", "_")
        return self.data_dir / f"{safe_name}_{self.timeframe}_raw.csv"

    def _get_checkpoint_ts(self, csv_path: Path) -> int:
        """
        체크포인트 타임스탬프 조회.
        CSV가 이미 존재하면 마지막 행의 timestamp(ms)를 반환하여
        이어받기 수집의 시작점으로 사용한다.
        파일이 없으면 최초 수집 시작점을 반환한다.
        """
        if csv_path.exists():
            try:
                # 파일 끝 부분만 효율적으로 읽기
                tail = pd.read_csv(csv_path, usecols=["timestamp"]).tail(1)
                if not tail.empty:
                    last_ts = int(tail["timestamp"].iloc[-1])
                    last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
                    logger.info(
                        "체크포인트 발견: %s → %s 이후부터 이어서 수집합니다.",
                        csv_path.name,
                        last_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    )
                    # 마지막 캔들 다음부터 수집 (중복 방지)
                    return last_ts + 1
            except Exception as e:
                logger.warning("체크포인트 읽기 실패 (%s), 처음부터 수집합니다. 오류: %s", csv_path.name, e)
        return self.start_ts

    def _fetch_ohlcv_chunk(self, symbol: str, since_ts: int) -> list[list]:
        """
        단일 API 호출로 OHLCV 데이터를 가져온다.
        네트워크 오류 시 최대 3회 재시도한다.

        Returns
        -------
        list of [timestamp, open, high, low, close, volume]
        """
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=self.timeframe,
                    since=since_ts,
                    limit=self.LIMIT_PER_REQUEST,
                )
                return candles
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                wait = 2 ** attempt  # 지수 백오프
                logger.warning(
                    "[%s] 네트워크 오류 (시도 %d/%d): %s — %d초 후 재시도",
                    symbol, attempt, max_retries, e, wait,
                )
                time.sleep(wait)
            except ccxt.ExchangeError as e:
                logger.error("[%s] 거래소 오류 (복구 불가): %s", symbol, e)
                raise
        raise RuntimeError(f"{symbol} 데이터 수집 실패: {max_retries}회 재시도 모두 실패")

    # ─────────────────────────────────────────
    # 심볼 단위 수집 메서드
    # ─────────────────────────────────────────

    def collect_symbol(self, symbol: str) -> Path:
        """
        단일 심볼의 전체 과거 데이터를 수집하여 CSV로 저장한다.

        - 체크포인트가 있으면 이어받기 수집
        - CHUNK_SIZE 단위로 청크를 쌓아 CSV에 Append 저장
        - 현재 시각을 넘으면 수집 종료

        Returns
        -------
        저장된 CSV 파일 경로
        """
        csv_path = self._get_csv_path(symbol)
        since_ts = self._get_checkpoint_ts(csv_path)
        now_ts = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        # CSV 존재 여부에 따라 헤더 포함 여부 결정
        write_header = not csv_path.exists()

        buffer: list[list] = []          # 청크 버퍼
        total_candles = 0

        logger.info("━━━ [%s] 수집 시작 ━━━", symbol)

        while since_ts < now_ts:
            candles = self._fetch_ohlcv_chunk(symbol, since_ts)

            if not candles:
                logger.info("[%s] 더 이상 수집할 데이터가 없습니다.", symbol)
                break

            buffer.extend(candles)
            since_ts = candles[-1][0] + 1  # 다음 요청 시작점 갱신

            # 청크가 가득 차면 CSV에 Append 저장 후 버퍼 비우기
            if len(buffer) >= self.CHUNK_SIZE:
                self._save_chunk(buffer, csv_path, write_header)
                total_candles += len(buffer)
                buffer.clear()
                write_header = False  # 두 번째 청크부터 헤더 미포함
                logger.info("[%s] 누적 저장: %d개 캔들", symbol, total_candles)

            time.sleep(self.SLEEP_BETWEEN_REQUESTS)

        # 버퍼에 남은 데이터 최종 저장
        if buffer:
            self._save_chunk(buffer, csv_path, write_header)
            total_candles += len(buffer)

        logger.info("━━━ [%s] 수집 완료 | 총 %d개 캔들 저장 ━━━", symbol, total_candles)
        return csv_path

    def _save_chunk(self, candles: list[list], csv_path: Path, write_header: bool) -> None:
        """
        캔들 리스트를 DataFrame으로 변환하여 CSV에 Append 저장한다.

        Parameters
        ----------
        candles      : [[timestamp, open, high, low, close, volume], ...]
        csv_path     : 저장 경로
        write_header : True면 헤더 행 포함 (최초 저장 시)
        """
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.to_csv(
            csv_path,
            mode="a",           # Append 모드
            header=write_header,
            index=False,
        )

    # ─────────────────────────────────────────
    # 전체 심볼 일괄 수집
    # ─────────────────────────────────────────

    def collect_all(self) -> dict[str, Path]:
        """
        self.symbols에 정의된 모든 심볼을 순차 수집한다.

        Returns
        -------
        {symbol: csv_path} 딕셔너리
        """
        result = {}
        for symbol in self.symbols:
            try:
                path = self.collect_symbol(symbol)
                result[symbol] = path
            except Exception as e:
                logger.error("[%s] 수집 중 치명적 오류 발생: %s — 다음 심볼로 넘어갑니다.", symbol, e)
        return result


# =============================================================================
# 2. 데이터 전처리 클래스
# =============================================================================
class CryptoDataPreprocessor:
    """
    수집된 원시 CSV 데이터를 머신러닝 학습에 적합한 형태로 전처리하는 클래스.
    - 결측치(NaN) 제거
    - TA-Lib 기반 기술적 지표 (MACD, RSI, ATR) 추가
    - Pandas 기반 TD Sequential 지표 구현
    - MinMaxScaler를 이용한 정규화
    """

    # 정규화 스케일러를 저장할 기본 경로
    SCALER_DIR: str = "data/scalers"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.scaler_dir = Path(self.SCALER_DIR)
        self.scaler_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────
    # 데이터 로드
    # ─────────────────────────────────────────

    def load_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        원시 CSV 파일을 로드하고 기본 정제를 수행한다.
        - timestamp 기준 중복 제거 및 정렬
        - datetime 컬럼을 인덱스로 설정
        """
        logger.info("CSV 로드: %s", csv_path)
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.set_index("datetime", inplace=True)
        df.index = df.index.tz_localize(None)  # tz-naive로 통일 (TA-Lib 호환)
        logger.info("로드 완료: %d행", len(df))
        return df

    # ─────────────────────────────────────────
    # 결측치 처리
    # ─────────────────────────────────────────

    def remove_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치(NaN)가 포함된 행을 제거한다.
        지표 계산 후 호출하면 warm-up 기간의 NaN도 함께 제거된다.
        """
        before = len(df)
        df.dropna(inplace=True)
        after = len(df)
        logger.info("결측치 제거: %d행 → %d행 (제거: %d행)", before, after, before - after)
        return df

    # ─────────────────────────────────────────
    # TA-Lib 기술적 지표 계산
    # ─────────────────────────────────────────

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence) 지표 추가.
        - macd       : MACD 라인 (12일 EMA - 26일 EMA)
        - macd_signal: 시그널 라인 (MACD의 9일 EMA)
        - macd_hist  : 히스토그램 (macd - macd_signal)
        """
        close = df["close"].values.astype(float)
        macd, signal, hist = talib.MACD(
            close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist
        logger.info("MACD 지표 추가 완료")
        return df

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        RSI (Relative Strength Index) 지표 추가.
        - rsi: 0~100 사이 값, 70 이상 과매수 / 30 이하 과매도

        Parameters
        ----------
        period : RSI 계산 기간 (기본값 14)
        """
        close = df["close"].values.astype(float)
        df["rsi"] = talib.RSI(close, timeperiod=period)
        logger.info("RSI(%d) 지표 추가 완료", period)
        return df

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        ATR (Average True Range) 지표 추가.
        - atr: 변동성을 절대값으로 표현하는 지표

        Parameters
        ----------
        period : ATR 계산 기간 (기본값 14)
        """
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        df["atr"] = talib.ATR(high, low, close, timeperiod=period)
        logger.info("ATR(%d) 지표 추가 완료", period)
        return df

    # ─────────────────────────────────────────
    # TD Sequential 지표 (Pandas 직접 구현)
    # ─────────────────────────────────────────

    def add_td_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TD Sequential 지표를 Pandas 연산으로 직접 구현하여 추가한다.

        TD Sequential은 Tom DeMark가 고안한 추세 전환 탐지 지표로,
        Setup 단계(1~9카운트)와 Countdown 단계로 구성된다.
        여기서는 가장 핵심적인 Setup 카운트를 구현한다.

        규칙:
        - 매수 Setup: 현재 종가 < 4봉 전 종가를 9번 연속 만족 → td_buy_setup (1~9)
        - 매도 Setup: 현재 종가 > 4봉 전 종가를 9번 연속 만족 → td_sell_setup (1~9)
        - 9에 도달하면 카운트 리셋 (새로운 Setup 탐색)

        추가 컬럼:
        - td_buy_setup  : 매수 Setup 카운트 (0이면 비활성)
        - td_sell_setup : 매도 Setup 카운트 (0이면 비활성)
        """
        close = df["close"].values
        n = len(close)

        buy_setup = np.zeros(n, dtype=int)
        sell_setup = np.zeros(n, dtype=int)

        buy_count = 0
        sell_count = 0

        for i in range(4, n):
            # ── 매수 Setup 조건: 현재 종가 < 4봉 전 종가 ──
            if close[i] < close[i - 4]:
                buy_count += 1
                sell_count = 0  # 반대 방향 카운트 리셋
            # ── 매도 Setup 조건: 현재 종가 > 4봉 전 종가 ──
            elif close[i] > close[i - 4]:
                sell_count += 1
                buy_count = 0   # 반대 방향 카운트 리셋
            else:
                # 조건 미충족: 양쪽 모두 리셋
                buy_count = 0
                sell_count = 0

            # 9카운트 도달 시 리셋 (값은 기록 후 리셋)
            buy_setup[i] = min(buy_count, 9)
            sell_setup[i] = min(sell_count, 9)

            if buy_count >= 9:
                buy_count = 0
            if sell_count >= 9:
                sell_count = 0

        df["td_buy_setup"] = buy_setup
        df["td_sell_setup"] = sell_setup
        logger.info("TD Sequential 지표 추가 완료")
        return df

    # ─────────────────────────────────────────
    # 전체 지표 추가 파이프라인
    # ─────────────────────────────────────────

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표를 순차적으로 추가하고 결측치를 제거한다.
        지표 계산 → NaN 제거 순서를 지켜야 warm-up 기간이 올바르게 제거된다.
        """
        df = self.add_macd(df)
        df = self.add_rsi(df)
        df = self.add_atr(df)
        df = self.add_td_sequential(df)
        df = self.remove_nulls(df)
        return df

    # ─────────────────────────────────────────
    # 정규화 (MinMaxScaler)
    # ─────────────────────────────────────────

    def normalize(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        symbol: str = "unknown",
    ) -> tuple[pd.DataFrame, MinMaxScaler]:
        """
        지정된 컬럼을 MinMaxScaler로 0~1 범위로 정규화한다.
        스케일러는 재사용(추론 시 역변환)을 위해 로컬에 저장한다.

        Parameters
        ----------
        df           : 전처리 완료된 DataFrame
        feature_cols : 정규화할 컬럼 목록. None이면 모든 수치 컬럼 대상
        symbol       : 스케일러 파일명 구분용 심볼 이름

        Returns
        -------
        (정규화된 DataFrame, 학습된 MinMaxScaler 객체)
        """
        if feature_cols is None:
            # timestamp 컬럼 제외, 수치형 컬럼 전체 선택
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "timestamp" in feature_cols:
                feature_cols.remove("timestamp")

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_norm = df.copy()
        df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])

        # 스케일러를 파일로 저장 (추후 역정규화 시 사용)
        safe_symbol = symbol.replace("/", "_")
        scaler_path = self.scaler_dir / f"{safe_symbol}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info("정규화 완료 | 스케일러 저장: %s", scaler_path)

        return df_norm, scaler

    # ─────────────────────────────────────────
    # 심볼 단위 전처리 통합 메서드
    # ─────────────────────────────────────────

    def process(
        self,
        csv_path: Path,
        symbol: str,
        save_processed: bool = True,
    ) -> tuple[pd.DataFrame, MinMaxScaler]:
        """
        단일 심볼의 전처리 파이프라인을 한 번에 실행한다.

        순서: 로드 → 지표 추가 → 결측치 제거 → 정규화

        Parameters
        ----------
        csv_path        : 원시 CSV 경로
        symbol          : 심볼 이름 (스케일러 저장명 및 로그용)
        save_processed  : True면 전처리 결과를 별도 CSV로 저장

        Returns
        -------
        (정규화된 DataFrame, MinMaxScaler)
        """
        logger.info("━━━ [%s] 전처리 시작 ━━━", symbol)
        df = self.load_csv(csv_path)
        df = self.add_all_indicators(df)
        df_norm, scaler = self.normalize(df, symbol=symbol)

        if save_processed:
            safe_name = symbol.replace("/", "_")
            out_path = self.data_dir / f"{safe_name}_processed.csv"
            df_norm.to_csv(out_path)
            logger.info("[%s] 전처리 완료 데이터 저장: %s", symbol, out_path)

        logger.info("━━━ [%s] 전처리 완료 | 최종 shape: %s ━━━", symbol, df_norm.shape)
        return df_norm, scaler


# =============================================================================
# 3. 스케일러 역정규화 유틸리티 (추론 시 활용)
# =============================================================================

def load_scaler(symbol: str, scaler_dir: str = "data/scalers") -> MinMaxScaler:
    """
    저장된 MinMaxScaler를 로드하여 반환한다.
    추론(Inference) 단계에서 모델 출력을 원래 가격 단위로 역변환할 때 사용한다.

    Parameters
    ----------
    symbol     : 심볼 이름 (예: "BTC/USDT")
    scaler_dir : 스케일러가 저장된 디렉토리

    Returns
    -------
    로드된 MinMaxScaler 객체
    """
    safe_symbol = symbol.replace("/", "_")
    path = Path(scaler_dir) / f"{safe_symbol}_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {path}")
    scaler = joblib.load(path)
    logger.info("스케일러 로드 완료: %s", path)
    return scaler


# =============================================================================
# 4. 실행 진입점
# =============================================================================

if __name__ == "__main__":
    # ── 수집 설정 ──────────────────────────────────────
    SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    TIMEFRAME = "5m"
    DATA_DIR = "data"
    LOOKBACK_YEARS = 3

    print("=" * 65)
    print("  암호화폐 딥러닝 트레이딩 봇 — 데이터 수집 및 전처리 파이프라인")
    print("=" * 65)

    # ── STEP 1: 과거 데이터 수집 ──────────────────────
    print("\n[STEP 1] 과거 데이터 수집 시작...")
    collector = CryptoDataCollector(
        symbols=SYMBOLS,
        timeframe=TIMEFRAME,
        data_dir=DATA_DIR,
        lookback_years=LOOKBACK_YEARS,
    )
    collected_paths = collector.collect_all()

    print("\n[STEP 1 완료] 수집된 파일 목록:")
    for sym, path in collected_paths.items():
        size_mb = path.stat().st_size / (1024 ** 2) if path.exists() else 0
        print(f"  • {sym:12s} → {path}  ({size_mb:.1f} MB)")

    # ── STEP 2: 전처리 (지표 추가 + 정규화) ──────────
    print("\n[STEP 2] 데이터 전처리 시작...")
    preprocessor = CryptoDataPreprocessor(data_dir=DATA_DIR)
    processed_results: dict[str, tuple[pd.DataFrame, MinMaxScaler]] = {}

    for symbol, csv_path in collected_paths.items():
        if not csv_path.exists():
            logger.warning("[%s] 원시 CSV가 없어 전처리를 건너뜁니다.", symbol)
            continue
        df_processed, scaler = preprocessor.process(
            csv_path=csv_path,
            symbol=symbol,
            save_processed=True,
        )
        processed_results[symbol] = (df_processed, scaler)

    # ── STEP 3: 결과 요약 출력 ─────────────────────────
    print("\n" + "=" * 65)
    print("  전처리 결과 요약")
    print("=" * 65)
    for symbol, (df, _) in processed_results.items():
        print(f"\n  [{symbol}]")
        print(f"    • 행 수       : {len(df):,}개")
        print(f"    • 컬럼        : {list(df.columns)}")
        print(f"    • 기간        : {df.index[0]} ~ {df.index[-1]}")
        print(f"    • 값 범위 확인: min={df['close'].min():.4f}, max={df['close'].max():.4f}")

    print("\n✅ 모든 작업이 완료되었습니다. 'data/' 폴더를 확인하세요.")
    print("=" * 65)