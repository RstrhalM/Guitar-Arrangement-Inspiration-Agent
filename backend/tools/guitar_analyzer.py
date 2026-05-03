import os
import json
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# ================= 基础配置与模板 =================
CHORD_TEMPLATES = {
    "Major": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
    "Minor": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
    "Dom7": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    "Maj7": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
    "m7": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),
    "Dim": np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
    "Aug": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
    "Sus4": np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
    "Sus2": np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
}

# 4-5类复杂和弦归并策略
CHORD_CATEGORY_MAP = {
    "Major": "Triad_Major", "Minor": "Triad_Minor",
    "Dom7": "Seventh_Extended", "Maj7": "Seventh_Extended", "m7": "Seventh_Extended",
    "Dim": "Altered_Dissonant", "Aug": "Altered_Dissonant", "Dim7": "Altered_Dissonant",
    "Sus4": "Suspended_Add", "Sus2": "Suspended_Add", "add9": "Suspended_Add"
}
DEFAULT_CATEGORY = "Triad_Base"


class GuitarArrangementAnalyzer:
    def __init__(self, sr=22050, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length
        self.tempo_threshold = 0.08  # 8%偏差视为变速
        self.beat_group_variance = 0.3  # 节拍分组方差阈值


    def _detect_tempo_meter_sections(self, y: np.ndarray) -> Dict:
        """librosa 节拍/速度/拍号分析 (按小节分组)"""
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr, hop_length=self.hop_length)

        # 计算局部BPM (每8个beat为一个检测窗)
        win_size = 8
        local_bpms = []
        for i in range(0, len(beat_times) - win_size + 1, win_size // 2):
            dt = beat_times[i + win_size - 1] - beat_times[i]
            if dt > 0: local_bpms.append((win_size - 1) / dt * 60)

        if len(local_bpms) < 2:
            return {"is_variable": False, "sections": [{"start_measure": 1, "end_measure": "ALL", "bpm": float(tempo)}],
                    "total_measures": 0}

        # 稳定性分割
        sections = []
        start_idx, current_bpm = 0, local_bpms[0]
        for i, bpm in enumerate(local_bpms):
            if abs(bpm - current_bpm) / current_bpm > self.tempo_threshold:
                sections.append({"start": start_idx, "end": i, "bpm": round(current_bpm, 1)})
                start_idx, current_bpm = i, bpm
        sections.append({"start": start_idx, "end": len(local_bpms), "bpm": round(current_bpm, 1)})

        # 拍号启发式推断 (基于beat间隔方差)
        meter_sections = []
        for sec in sections:
            s, e, bpm = sec["start"], sec["end"], sec["bpm"]
            beat_ints = np.diff(beat_times[s * 4:e * 4])  # 粗略按4/4对齐
            var = np.var(beat_ints) if len(beat_ints) > 0 else 0
            meter = "4/4" if var < self.beat_group_variance else "3/4"
            meter_sections.append({"meter": meter})

        # 合并并计算小节数 (假设4/4默认，按BPM换算时长)
        total_beats = len(beat_times)
        measures_per_sec = [s["end"] - s["start"] for s in sections]
        # 简化：按全局平均beat间隔估算总小节
        avg_beat_dur = np.mean(np.diff(beat_times)) if len(beat_times) > 1 else 0.5
        total_measures = int(total_beats * avg_beat_dur / (4 * (60 / float(tempo)))) + 1

        final_secs = []
        cur_m = 1
        for i, sec in enumerate(sections):
            m_count = max(1, int((sec["end"] - sec["start"]) * 0.5))  # 启发式映射
            final_secs.append({
                "start_measure": cur_m,
                "end_measure": cur_m + m_count - 1,
                "bpm": sec["bpm"],
                "meter": meter_sections[i]["meter"]
            })
            cur_m += m_count

        return {
            "is_variable": len(sections) > 1,
            "sections": final_secs,
            "total_measures": cur_m - 1,
            "overall_analysis": "变拍变速已检测" if len(sections) > 1 else "全曲节拍速度稳定"
        }

    def _analyze_harmony(self, midi_path: str, tempo_sections: List[Dict]) -> Dict:
        """基于 pretty_midi 的段落和弦与调性分析 (MIDI原生驱动，彻底移除librosa音频特征)"""
        pm = pretty_midi.PrettyMIDI(midi_path)
        # fs=50 足够捕获和声变化，返回形状 (128, T)
        piano_roll = pm.get_piano_roll(fs=50)
        n_frames = piano_roll.shape[1]

        # 1. MIDI 128键 → 12音级 Chroma 映射
        chroma = np.zeros((12, n_frames))
        for midi_note in range(128):
            pc = midi_note % 12
            chroma[pc, :] += piano_roll[midi_note, :]

        # 逐帧归一化，消除力度差异干扰
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        chroma = chroma / norms

        harmony_out = []
        keys = []
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        major_prof = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_prof = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # 2. 按小节数比例安全切分段落帧范围
        total_measures = sum(sec["end_measure"] - sec["start_measure"] + 1 for sec in tempo_sections)
        frames_per_measure = n_frames / max(1, total_measures)
        current_frame = 0

        for sec in tempo_sections:
            n_m = sec["end_measure"] - sec["start_measure"] + 1
            sec_len = int(n_m * frames_per_measure)
            start_idx = current_frame
            end_idx = min(start_idx + sec_len, n_frames)
            current_frame = end_idx

            if end_idx <= start_idx:
                harmony_out.append(
                    {"measure_range": [sec["start_measure"], sec["end_measure"]], "key": "Unknown", "chords": []})
                continue

            sec_chroma = chroma[:, start_idx:end_idx]
            sec_mean = np.mean(sec_chroma, axis=1)

            # 3. 调性检测 (Krumhansl-Schmuckler)
            best_corr, best_key = -1.0, "C Major"
            for root in range(12):
                c_maj = np.dot(sec_mean, np.roll(major_prof, root))
                c_min = np.dot(sec_mean, np.roll(minor_prof, root))
                if c_maj > best_corr: best_corr, best_key = c_maj, f"{note_names[root]} Major"
                if c_min > best_corr: best_corr, best_key = c_min, f"{note_names[root]} Minor"
            keys.append(best_key)

            # 4. 和弦匹配 (每小节采样1次)
            chords = []
            frames_per_bar = max(1, int(frames_per_measure))
            for i in range(0, sec_chroma.shape[1], frames_per_bar):
                bar_vec = np.mean(sec_chroma[:, i:i + frames_per_bar], axis=1)
                best_sim, best_name, best_cat = 0.0, "N.C.", DEFAULT_CATEGORY

                for cname, tpl in CHORD_TEMPLATES.items():
                    for root in range(12):
                        tpl_roll = np.roll(tpl, root)
                        sim = np.dot(bar_vec, tpl_roll) / (np.linalg.norm(bar_vec) * np.linalg.norm(tpl_roll) + 1e-6)
                        if sim > best_sim:
                            best_sim, best_name, best_cat = sim, f"{note_names[root]}{cname}", CHORD_CATEGORY_MAP.get(
                                cname, DEFAULT_CATEGORY)
                chords.append({"chord": best_name, "category": best_cat})

            harmony_out.append({
                "measure_range": [sec["start_measure"], sec["end_measure"]],
                "key": best_key,
                "chords": chords
            })

        return {"key_changes": len(set(keys)) > 1, "sections": harmony_out}

    def _clean_distorted_strum_midi(self, midi_path: str,
                                    quantize_grid: float = 0.0625,  # 量化网格(秒)，默认1/32音符
                                    merge_window: float = 0.025,  # 扫弦合并窗口(秒)
                                    min_vel: int = 25) -> None:
        """专为失真扫弦设计的 MIDI 后处理：量化时序 + 合并扫弦簇 + 过滤泛音噪声"""
        pm = pretty_midi.PrettyMIDI(midi_path)
        for inst in pm.instruments:
            notes = sorted(inst.notes, key=lambda n: n.start)
            cleaned = []

            i = 0
            while i < len(notes):
                # 1. 扫弦时间窗合并
                cluster = [notes[i]]
                j = i + 1
                while j < len(notes) and notes[j].start - notes[i].start < merge_window:
                    cluster.append(notes[j])
                    j += 1

                # 2. 量化对齐
                for note in cluster:
                    note.start = round(note.start / quantize_grid) * quantize_grid
                    note.end = max(note.start + 0.04, round(note.end / quantize_grid) * quantize_grid)

                # 3. 泛音过滤（失真扫弦典型问题：中间音多为物理泛音或串音）
                if len(cluster) >= 3:
                    cluster.sort(key=lambda n: n.pitch)
                    # 保留：根音(最低) + 旋律音(最高) + 力度最大的1个中间音
                    kept = [cluster[0], cluster[-1]]
                    if len(cluster) > 3:
                        mid_sorted = sorted(cluster[1:-1], key=lambda n: n.velocity, reverse=True)
                        kept.append(mid_sorted[0])
                    cleaned.extend(kept)
                else:
                    cleaned.extend(cluster)
                i = j

            # 4. 二次过滤：极短/极低力度残响
            inst.notes = [n for n in cleaned if (n.end - n.start) >= 0.04 and n.velocity >= min_vel]

        pm.write(midi_path)

    def _filter_silent_notes(self, audio_path: str, midi_path: str,
                             energy_threshold_db: float = -30.0,
                             min_velocity: int = 15,
                             min_duration: float = 0.01) -> None:
        """基于音频能量包络过滤静音段误检音符与极短幻觉音"""
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        hop_length = 256  # 高时间分辨率对齐
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms))
        times = librosa.frames_to_time(range(len(rms_db)), sr=sr, hop_length=hop_length)

        # 能量掩码：低于峰值 threshold_db dB 视为静音区
        energy_mask = rms_db > energy_threshold_db

        pm = pretty_midi.PrettyMIDI(midi_path)
        for inst in pm.instruments:
            filtered_notes = []
            for note in inst.notes:
                # 1. 基础过滤：力度过低或时长过短（BP典型幻觉特征）
                vel = getattr(note, 'velocity', 64)
                if vel < min_velocity or (note.end - note.start) < min_duration:
                    continue

                # 2. 音频能量时空对齐
                s_idx = np.searchsorted(times, note.start, side='left')
                e_idx = np.searchsorted(times, note.end, side='right')
                s_idx = np.clip(s_idx, 0, len(energy_mask) - 1)
                e_idx = np.clip(e_idx, 0, len(energy_mask))

                if e_idx > s_idx:
                    energy_ratio = np.mean(energy_mask[s_idx:e_idx])
                else:
                    energy_ratio = 0.0

                # 保留条件：音符覆盖区间内 ≥40% 时间存在有效音频能量
                if energy_ratio >= 0.4:
                    filtered_notes.append(note)

            inst.notes = filtered_notes
        pm.write(midi_path)

    def _preprocess_for_basic_pitch(self, audio_path: str) -> str:
        """针对失真/过载吉他音频的预处理：频域清洗 + 采样率对齐 + 动态归一化"""
        import scipy.signal as signal
        import scipy.io.wavfile as wavfile
        import tempfile
        import os

        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # 1. 强制重采样至 22050Hz (basic_pitch 原生训练采样率，不匹配会严重降质)
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050

        # 2. 4阶巴特沃斯带通滤波 (80Hz ~ 5000Hz)
        # 切除低频轰鸣与高频失真“嘶声”，保留吉他有效谐波骨架
        nyq = sr / 2.0
        b, a = signal.butter(4, [80 / nyq, 5000 / nyq], btype='band')
        y = signal.filtfilt(b, a, y)

        # 3. 峰值归一化至 -3dBFS (防止神经网因削波饱和产生幻觉长音)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y * (0.707 / peak)

        # 保存为临时 WAV 供 basic_pitch 读取（兼容所有版本）
        tmp_path = os.path.join(tempfile.gettempdir(), f"bp_pre_{os.path.basename(audio_path)}.wav")
        wavfile.write(tmp_path, sr, (y * 32767).astype(np.int16))
        return tmp_path

    def _audio_to_midi(self, preprocessed_wav: str, midi_out: str) -> str:
        """接收预处理后的 WAV 文件，输出 MIDI"""
        try:
            from basic_pitch.inference import predict
            # basic_pitch 最新版支持直接传路径或 (audio, sr) 元组
            result = predict(preprocessed_wav)

            # 动态提取 MIDI 对象（兼容多版本返回结构）
            midi_data = next((item for item in result if hasattr(item, 'write')), result[1])
            midi_data.write(midi_out)
            return midi_out
        except Exception as e:
            raise RuntimeError(f"Basic Pitch 转换失败: {e}")

    def run(self, audio_path: str, midi_out: Optional[str] = None) -> Dict:
        audio_path = str(Path(audio_path).resolve())
        if midi_out is None:
            midi_out = Path(audio_path).with_suffix(".mid").name

        print("[1/5] 音频预处理(失真频段清洗+重采样)...")
        pre_wav = self._preprocess_for_basic_pitch(audio_path)

        print(f"[2/5] 音频转MIDI: {Path(midi_out).name}")
        self._audio_to_midi(pre_wav, midi_out)
        os.remove(pre_wav)  # 清理临时文件

        print("[3/5] 静音段幻觉过滤...")
        self._filter_silent_notes(audio_path, midi_out)

        print("[4/5] 失真扫弦清理(量化+合并+泛音过滤)...")
        self._clean_distorted_strum_midi(midi_out)

        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)

        print("[5/5] 节拍/和声分析...")
        rhythm_data = self._detect_tempo_meter_sections(y)
        harmony_data = self._analyze_harmony(midi_out, rhythm_data["sections"])

        return {
            "metadata": {"file": audio_path, "midi": midi_out, "total_measures": rhythm_data["total_measures"]},
            "tempo_structure": rhythm_data,
            "harmonic_analysis": harmony_data,
            "summary": f"全曲{rhythm_data['total_measures']}小节，{'包含' if rhythm_data['is_variable'] else '无'}变速/变拍。"
                       f"和声{'包含转调' if harmony_data['key_changes'] else '调性统一'}。"
        }

    def save_json(self, result: Dict, out_path: str):
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    analyzer = GuitarArrangementAnalyzer()
    res = analyzer.run("inhale.mp3")
    analyzer.save_json(res, "analysis_result.json")
    print("✅ 分析完成，JSON已生成。")