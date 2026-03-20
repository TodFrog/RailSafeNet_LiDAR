"""`videoAssessor` 통합 추론에서 재사용하는 선로 보조 함수 모음.

이 모듈은 `001-what-why-home` branch의 `videoAssessor.py`에 있던 핵심 선로 추정
함수만 추려서 현재 저장소 구조에 맞게 정리한 것이다. 최종 활성 런타임은
`videoAssessor.py` 하나로 통일하지만, 선로 후보 추정과 비디오 선택 로직은 별도
모듈로 분리해 두는 편이 유지보수와 delivery review에 유리하다.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np


def find_extreme_y_values(arr: np.ndarray, values: Iterable[int] = (4, 9)) -> tuple[int | None, int | None]:
    """지정 클래스가 처음/마지막으로 등장하는 y 좌표를 찾는다.

    현재 파이프라인은 클래스 ID `4`, `9`를 선로 후보로 사용한다. 이 매핑의 최종
    근거는 문서 기준으로 `TODO` 상태지만, 기존 모델과 후처리 로직이 이 가정을
    공유하고 있으므로 동작 보존을 위해 그대로 유지한다.
    """
    mask = np.isin(arr, list(values))
    rows_with_values = np.any(mask, axis=1)
    y_indices = np.nonzero(rows_with_values)[0]
    if y_indices.size == 0:
        return None, None
    return int(y_indices[0]), int(y_indices[-1])


def find_edges(image: np.ndarray, y_levels: list[int], values: Iterable[int] = (4, 9), min_width: int = 19) -> dict[int, list[tuple[int, int]]]:
    """각 y 레벨에서 선로 클래스의 연속 구간을 찾는다.

    `min_width=19`는 너무 얇은 오검출 띠를 제거하기 위한 휴리스틱이다. 기존
    `001-what-why-home` branch의 최신 `videoAssessor.py`가 같은 값을 사용하므로
    여기서도 그대로 유지한다.
    """
    edges_dict: dict[int, list[tuple[int, int]]] = {}
    for y in y_levels:
        if y >= image.shape[0]:
            continue
        row = image[y, :]
        mask = np.isin(row, list(values)).astype(int)
        diff = np.diff(np.pad(mask, (1, 1), "constant"))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        filtered_edges = [
            (int(start), int(end))
            for start, end in zip(starts, ends)
            if end - start + 1 >= min_width and start != 0 and end != image.shape[1] - 1
        ]
        if filtered_edges:
            edges_dict[int(y)] = filtered_edges
    return edges_dict


def identify_ego_track(
    edges_dict: dict[int, list[tuple[int, int]]],
    image_width: int,
    previous_track_center: float | None = None,
) -> dict[int, list[tuple[int, int]]]:
    """탐지된 후보 중 실제 ego track에 가장 가까운 구간을 선택한다.

    첫 프레임에서는 화면 중심과 가장 가까운 선로를 고르고, 이전 프레임 중심이
    있으면 그 값을 우선 참조한다. 이는 001 branch의 최신 helper가 사용한 temporal
    consistency 전략과 동일하다.
    """
    ego_edges_dict: dict[int, list[tuple[int, int]]] = {}
    last_ego_track_center: float | None = None
    sorted_y_levels = sorted(edges_dict.keys(), reverse=True)
    reference_center = previous_track_center if previous_track_center is not None else image_width / 2

    if sorted_y_levels:
        first_y = sorted_y_levels[0]
        tracks_at_first_y = edges_dict.get(first_y, [])
        if tracks_at_first_y:
            closest_track = min(
                tracks_at_first_y,
                key=lambda track: abs(((track[0] + track[1]) / 2) - reference_center),
            )
            ego_edges_dict[first_y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2

    for y in sorted_y_levels[1:]:
        if last_ego_track_center is None:
            break
        tracks_at_y = edges_dict.get(y, [])
        if not tracks_at_y:
            continue
        closest_track = min(
            tracks_at_y,
            key=lambda track: abs(((track[0] + track[1]) / 2) - last_ego_track_center),
        )
        ego_edges_dict[y] = [closest_track]
        last_ego_track_center = (closest_track[0] + closest_track[1]) / 2

    return ego_edges_dict


def find_rails(arr: np.ndarray, y_levels: list[int], values: Iterable[int] = (4, 9), min_width: int = 5) -> list[tuple[int, int]]:
    """좁은 rail strip까지 포함해 실제 레일 경계 후보를 재검색한다."""
    filtered_edges: list[tuple[int, int]] = []
    for y in y_levels:
        if y >= arr.shape[0]:
            continue
        row = arr[y, :]
        mask = np.isin(row, list(values)).astype(int)
        diff = np.diff(np.pad(mask, (1, 1), "constant"))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        filtered_edges = [
            (int(start), int(end))
            for start, end in zip(starts, ends)
            if end - start + 1 >= min_width and start != 0 and end != arr.shape[1] - 1
        ]
    return filtered_edges


def find_rail_sides(img: np.ndarray, edges_dict: dict[int, list[tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray]:
    """ego track 가장자리와 실제 rail strip을 맞춰 좌우 경계를 보정한다."""
    left_border: list[list[int]] = []
    right_border: list[list[int]] = []

    for y, xs in edges_dict.items():
        rails = find_rails(img, [y], values=(4, 9), min_width=5)
        left_border_actual = [min(xs)[0], y]
        right_border_actual = [max(xs)[1], y]
        for zone in rails:
            if abs(zone[1] - left_border_actual[0]) < y * 0.04:
                left_border_actual[0] = zone[0]
            if abs(zone[0] - right_border_actual[0]) < y * 0.04:
                right_border_actual[0] = zone[1]
        left_border.append(left_border_actual)
        right_border.append(right_border_actual)

    if len(left_border) > 2:
        left_arr = np.array(left_border)
        left_border = left_arr[left_arr[:, 1] != left_arr[:, 1].max()].tolist()
    if len(right_border) > 2:
        right_arr = np.array(right_border)
        right_border = right_arr[right_arr[:, 1] != right_arr[:, 1].max()].tolist()

    return np.array(left_border), np.array(right_border)


def get_clues(segmentation_mask: np.ndarray, number_of_clues: int) -> list[int]:
    """이미지 하단 45% 구간만 대상으로 y sampling 위치를 생성한다."""
    height = segmentation_mask.shape[0]
    start_y = int(height * 0.55)
    limited_mask = segmentation_mask[start_y:, :]

    lowest, highest = find_extreme_y_values(limited_mask)
    if lowest is None or highest is None or highest <= lowest:
        return []

    actual_lowest = lowest + start_y
    actual_highest = highest + start_y
    clue_step = int((actual_highest - actual_lowest) / (number_of_clues + 1))
    if clue_step == 0:
        clue_step = 1
    return [int(actual_highest - (i * clue_step)) for i in range(number_of_clues)] + [int(actual_lowest)]


def natural_sort_key(value: str) -> list[int | str]:
    """파일명을 사람 눈에 자연스럽게 정렬하기 위한 key."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", value)]


def select_video_file(video_dir: str | os.PathLike[str]) -> str | None:
    """비디오 파일이 여러 개 있을 때 사용자에게 선택 기회를 제공한다.

    현재 저장소는 샘플 비디오를 기본 포함하지 않으므로, 이 함수는 `--video`를
    주지 않은 경우에만 보조적으로 사용된다.
    """
    base_dir = Path(video_dir)
    if base_dir.is_file():
        return str(base_dir)
    if not base_dir.exists():
        print(f"비디오 디렉터리를 찾지 못했습니다: {base_dir}")
        return None

    patterns = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for pattern in patterns:
        video_files.extend(base_dir.rglob(pattern))
    video_files = sorted({path.resolve() for path in video_files}, key=lambda path: natural_sort_key(path.name))

    if not video_files:
        print(f"선택 가능한 비디오 파일이 없습니다: {base_dir}")
        return None

    print("사용 가능한 비디오 파일 목록:")
    for index, video_path in enumerate(video_files, start=1):
        print(f"{index}: {video_path.name}")

    while True:
        choice = input(f"비디오 번호를 선택하세요 (1-{len(video_files)}) 또는 'q': ").strip()
        if choice.lower() == "q":
            return None
        try:
            selected_index = int(choice) - 1
        except ValueError:
            print("숫자 또는 'q'만 입력할 수 있습니다.")
            continue
        if 0 <= selected_index < len(video_files):
            return str(video_files[selected_index])
        print(f"1부터 {len(video_files)} 사이의 번호를 입력하세요.")
