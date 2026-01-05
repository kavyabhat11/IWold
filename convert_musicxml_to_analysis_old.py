#!/usr/bin/env python3
"""
Convert between MusicXML harmonic analysis and plain text format.

Usage:
    # MusicXML to text:
    python convert_musicxml_to_analysis.py <input.musicxml> [output.txt]

    # Text to MusicXML:
    python convert_musicxml_to_analysis.py <input.txt> [output.musicxml]
"""

import xml.etree.ElementTree as ET
import sys
import re
from typing import List, Tuple, Optional


# ======================================================================
# FIGURED BASS TRANSFORMATIONS
# ======================================================================

def transform_figured_bass(harmony: str) -> str:
    """Convert MusicXML-style figured bass (64, 65, 43, etc.) into slash form."""
    harmony = harmony.replace('ø', '/o')

    patterns = [
        (r'(\w+?)65\b', r'\g<1>6/5'),
        (r'(\w+?)43\b', r'\g<1>4/3'),
        (r'(\w+?)42\b', r'\g<1>4/2'),
        (r'(\w+?)64\b', r'\g<1>6/4'),
        (r'(\w+?)32\b', r'\g<1>3/2'),
    ]

    for pattern, repl in patterns:
        harmony = re.sub(pattern, repl, harmony)

    return harmony


def reverse_transform_figured_bass(harmony: str) -> str:
    """Convert slash form (6/5, 4/3) back into MusicXML-style figured bass."""
    patterns = [
        (r'(\w+?)6/5\b', r'\g<1>65'),
        (r'(\w+?)4/3\b', r'\g<1>43'),
        (r'(\w+?)4/2\b', r'\g<1>42'),
        (r'(\w+?)6/4\b', r'\g<1>64'),
        (r'(\w+?)3/2\b', r'\g<1>32'),
    ]

    for pattern, repl in patterns:
        harmony = re.sub(pattern, repl, harmony)

    harmony = harmony.replace('/o', 'ø')
    return harmony


# ======================================================================
# BEAT / OFFSET HANDLERS
# ======================================================================

def duration_to_beat(duration: int, divisions: int) -> Tuple[int, Optional[float]]:
    beat_offset = duration / divisions
    beat_number = int(beat_offset) + 1
    remainder = beat_offset - int(beat_offset)
    return (beat_number, remainder if remainder > 0.001 else None)


def format_beat(beat: int, subdivision: Optional[float]) -> str:
    return f"b{beat}" if subdivision is None else f"b{beat + subdivision}"


def beat_to_duration(beat_str: str, divisions: int = 10080) -> int:
    beat_value = float(beat_str.lstrip('b'))
    beat_offset = beat_value - 1
    return int(beat_offset * divisions)


# ======================================================================
# PARSE ROMAN-TEXT (.txt) → structured dict
# ======================================================================

def parse_text_analysis(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    metadata = {}
    measures = []
    comments = []
    current_key = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '[TODO' in line:
            continue

        # Metadata line
        if ':' in line and not line.startswith('m'):
            k, v = line.split(':', 1)
            k = k.strip()
            v = v.strip()
            if k in ['Composer', 'Piece', 'Analyst', 'Proofreader', 'Tempo', 'Time Signature']:
                metadata[k] = v
            elif k in ['Form', 'Pedal', 'Note']:
                comments.append(line)
            continue

        # Measure line
        if line.startswith('m'):
            if '=' in line:  # equivalence
                comments.append(line)
                continue

            parts = line.split()
            if not parts:
                continue

            measure_num = parts[0][1:]
            if 'var' in measure_num:
                comments.append(line)
                continue

            harmonies = []
            i = 1
            current_beat = None

            while i < len(parts):
                part = parts[i]

                if part.startswith('b'):
                    current_beat = part
                    i += 1
                    continue
                if part.endswith(':'):
                    current_key = part[:-1]
                    i += 1
                    continue

                # Harmony
                harmony = part
                if current_beat is None:
                    current_beat = 'b1'
                harmonies.append((current_beat, current_key, harmony))
                current_beat = None
                i += 1

            measures.append({
                "number": measure_num,
                "harmonies": harmonies
            })

    return {
        "metadata": metadata,
        "measures": measures,
        "comments": comments
    }


# ======================================================================
# PARSE MUSICXML → RomanText lines
# ======================================================================

def parse_musicxml(filepath: str) -> Tuple[dict, List[str]]:
    """Return (metadata, analysis_lines) extracted from MusicXML."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    ns_prefix = ''
    if '}' in root.tag:
        ns = root.tag.split('}')[0].strip('{')
        ns_prefix = '{' + ns + '}'

    # Find RNA part
    rna_part_id = None
    for part_list in root.findall(f'.//{ns_prefix}score-part'):
        pid = part_list.get('id')
        pname = part_list.find(f'{ns_prefix}part-name')
        if pid == 'RNA':
            rna_part_id = pid
            break
        if pname is not None and pname.text:
            t = pname.text.strip()
            if 'Roman' in t or 'RNA' in t or 'Numeral' in t:
                rna_part_id = pid
                break

    if not rna_part_id:
        raise ValueError("No RNA part found.")

    rna_part = None
    for p in root.findall(f'.//{ns_prefix}part'):
        if p.get('id') == rna_part_id:
            rna_part = p
            break

    if rna_part is None:
        raise ValueError("RNA part missing.")

    # divisions
    divisions_elem = rna_part.find(f'.//{ns_prefix}divisions')
    divisions = int(divisions_elem.text) if divisions_elem is not None else 10080

    # time signature
    ts_elem = rna_part.find(f'.//{ns_prefix}time')
    if ts_elem is not None:
        beats_elem = ts_elem.find(f'{ns_prefix}beats')
        beat_type_elem = ts_elem.find(f'{ns_prefix}beat-type')
        if beats_elem is not None and beat_type_elem is not None:
            parsed_time_sig = beats_elem.text.strip() + "/" + beat_type_elem.text.strip()
        else:
            parsed_time_sig = "4/4"
    else:
        parsed_time_sig = "4/4"

    metadata = {"Time Signature": parsed_time_sig}

    # Extract measures
    analysis_lines = []
    current_key = None

    for measure in rna_part.findall(f'{ns_prefix}measure'):
        number = measure.get('number')
        current_position = 0
        harmonies = []

        for elem in measure:
            tag = elem.tag.replace(ns_prefix, '')

            if tag == 'forward':
                dur = elem.find(f'{ns_prefix}duration')
                if dur is not None:
                    current_position += int(dur.text)

            elif tag == 'harmony':
                fx = elem.find(f'{ns_prefix}function')
                if fx is not None and fx.text:
                    harmonies.append((current_position, fx.text.strip()))

            elif tag == 'note':
                dur = elem.find(f'{ns_prefix}duration')
                if dur is not None:
                    current_position += int(dur.text)

        if not harmonies:
            continue

        out = [f"m{number}"]
        measure_key = None

        for i, (pos, harm) in enumerate(harmonies):

            # Key changes
            if ':' in harm:
                key, h2 = harm.split(':', 1)
                is_new = key != measure_key

                if i == 0:
                    if pos > 0:
                        beat, sub = duration_to_beat(pos, divisions)
                        out.append(format_beat(beat, sub))
                    out.append(f"{key}:")
                elif is_new:
                    beat, sub = duration_to_beat(pos, divisions)
                    out.append(format_beat(beat, sub))
                    out.append(f"{key}:")
                else:
                    beat, sub = duration_to_beat(pos, divisions)
                    out.append(format_beat(beat, sub))

                measure_key = key
                harm = h2
            else:
                if i == 0:
                    if pos > 0:
                        beat, sub = duration_to_beat(pos, divisions)
                        out.append(format_beat(beat, sub))
                    if current_key and current_key != measure_key:
                        out.append(f"{current_key}:")
                        measure_key = current_key
                else:
                    beat, sub = duration_to_beat(pos, divisions)
                    out.append(format_beat(beat, sub))

            # transform figured bass
            harm = transform_figured_bass(harm)
            out.append(harm)

        analysis_lines.append(" ".join(out))

    return metadata, analysis_lines


# ======================================================================
# GENERATE TEXT OUTPUT
# ======================================================================

def generate_output(metadata: dict, analysis_lines: List[str]) -> str:
    output = []
    output.append("Composer: [TODO]")
    output.append("Piece: [TODO]")
    output.append("Analyst: [TODO]")
    output.append("Proofreader: [TODO]")
    output.append("Note: [TODO]")
    output.append("")
    output.append("Tempo: [TODO]")
    output.append(f"Time Signature: {metadata.get('Time Signature', '4/4')}")
    output.append("")
    output.append("Form: [TODO]")
    output.append("[TODO: Add pedal markings if applicable]")
    output.append("")

    output.extend(analysis_lines)

    output.append("")
    output.append("[TODO Review and add as needed]")
    return "\n".join(output)


# ======================================================================
# MAIN
# ======================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: convert_musicxml_to_analysis.py <input> [output]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    is_xml = input_file.endswith(".xml") or input_file.endswith(".musicxml")
    is_txt = input_file.endswith(".txt")

    if not (is_xml or is_txt):
        print("Error: expected .xml/.musicxml or .txt")
        sys.exit(1)

    try:
        if is_xml:
            print(f"Converting MusicXML → text: {input_file}")
            metadata, analysis_lines = parse_musicxml(input_file)
            output = generate_output(metadata, analysis_lines)

            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output)
                print(f"Written: {output_file}")
            else:
                print(output)

        else:
            print(f"Converting text → MusicXML: {input_file}")
            data = parse_text_analysis(input_file)
            xml_str = generate_musicxml(data)

            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(xml_str)
                print(f"Written: {output_file}")
            else:
                print(xml_str)

    except Exception as e:
        print("\nERROR:", e, file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
