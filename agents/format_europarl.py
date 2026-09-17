"""
Transforma archivos Europarl crudos al formato WikiText con 3 niveles de encabezados.

Formato de entrada (ep-99-12-16.raw):
    = Session Title =
    = = Speaker = =
    Speech text...
    Topic Name
    = = President = =

Formato de salida (como ep-99-12-17.raw):
    = Session Title =
     = = Topic Name = =
     = = = Speaker = = =
    Speech text...

Transformaciones:
    1. Speakers  = = Name = =  →  = = = Name = = =  (nivel 2 → nivel 3)
    2. Topic lines (texto plano) →  = = Topic = =  (nuevo nivel 2)
    3. Long report/resolution titles at topic boundaries → kept as topic headers

Uso:
    python3 format_europarl.py data/Europarl/ep-99-12-16.raw
    python3 format_europarl.py data/Europarl/ep-99-12-16.raw --output data/Europarl/ep-99-12-16_formatted.raw
    python3 format_europarl.py data/Europarl/ep-99-12-16.raw --dry-run
"""

import argparse
import re
from pathlib import Path


# ─── Patrones ─────────────────────────────────────────────────────────────────

SPEAKER_RE = re.compile(r'^(\s*)= = (.+?) = =(\s*)$')
L1_RE = re.compile(r'^\s*= [^=].+[^=] =\s*$')
PROCEDURAL_RE = re.compile(
    r'^\('
    r'(Parliament|The sitting|The President|Applause|Loud applause|'
    r'The \d+ General|Interruption|Laughter)'
)

# Lines that are clearly NOT topics (speech content)
SPEECH_STARTERS = [
    'Mr ', 'Mrs ', 'Ms ', 'Madam ', 'Sir ', 'Lord ',
    'Thank ', 'I ', 'We ', 'You ', 'He ', 'She ', 'They ',
    'My ', 'Our ', 'Your ', 'His ', 'Her ', 'Their ', 'Its ',
    'The Minutes', 'The Commission', 'The Council', 'The debate',
    'The next item', 'The vote', 'The joint', 'The sitting',
    'This ', 'That ', 'These ', 'Those ',
    'Are ', 'Is ', 'Do ', 'Does ', 'Did ', 'Will ', 'Would ', 'Can ', 'Could ',
    'May I', 'Let ', 'In my', 'In this', 'In the', 'In any',
    'On this', 'On behalf', 'On a ',
    'For ', 'As ', 'Since ', 'After ', 'Although ', 'Before ',
    'However', 'Indeed', 'Nevertheless', 'Furthermore', 'Moreover',
    'First', 'Second', 'Third', 'Finally', 'Also',
    'Commissioner', 'Ladies', 'Absolutely',
    'Like ', 'Once ', 'Just ', 'Now ', 'Here ', 'There ',
    'It ', 'It\'s ', 'What ', 'When ', 'Where ', 'Why ', 'How ',
    'Not ', 'No ', 'Yes ',
    'With ', 'Without ', 'Despite ', 'During ',
    'Such ', 'Some ', 'Any ', 'All ', 'Both ', 'Each ', 'Every ',
    'If ', 'But ', 'So ', 'Yet ', 'And ', 'Or ',
    'Having ', 'Being ', 'Given ',
    '...', '- (', '. (', '- The', '- Mr', '- We', '- In',
    'Strictly', 'Certainly', 'Obviously', 'Unfortunately', 'Sadly',
    'To ', 'At ', 'By ', 'From ',
    'A ', 'An ',
    'While ', 'Whilst ', 'Whether ',
    'Even ', 'Only ', 'Perhaps ', 'Possibly ',
    'Relations ',  # "Relations with Turkey..."
]

# Lines that should never be topics
NEVER_TOPIC_RE = re.compile(
    r'^(B5-|'                           # resolution references
    r'\(|'                              # procedural notes
    r'-\s|'                             # bullet points
    r'\d+\.\s|'                         # numbered lists
    r'[a-z])'                           # starts with lowercase
)


def is_topic_line(line: str, prev_line: str, next_line: str,
                  prev2_line: str = '') -> bool:
    """Determine if a standalone line is a topic title."""
    s = line.strip()

    # Basic exclusions
    if not s or len(s) < 3 or len(s) > 250:
        return False
    if SPEAKER_RE.match(s) or L1_RE.match(s):
        return False
    if NEVER_TOPIC_RE.match(s):
        return False
    if any(s.startswith(p) for p in SPEECH_STARTERS):
        return False

    # Context: should be at a topic boundary
    prev_s = prev_line.strip()
    next_s = next_line.strip()
    prev2_s = prev2_line.strip()

    # Boundary indicators: previous line is procedural or empty after procedural
    at_boundary = (
        not prev_s  # blank line before
        or PROCEDURAL_RE.match(prev_s)
        or (not prev_s and PROCEDURAL_RE.match(prev2_s))
    )

    # Key pattern: topic line followed by blank line then speaker header
    # e.g., "Topic Name\n\n = = Speaker = ="
    # This is the most common pattern for inline topics
    followed_by_blank_then_speaker = (
        not next_s  # next line is blank
    )

    # Also consider: line is followed directly by a speaker header
    followed_by_speaker = SPEAKER_RE.match(next_s)

    if not at_boundary and not followed_by_speaker and not followed_by_blank_then_speaker:
        return False

    # Additional checks for false positives
    # Exclude lines that look like continuation sentences
    if s.endswith(',') or s.endswith(';'):
        return False

    # Lines that are clearly topic titles
    if len(s) < 80:
        return True

    # Longer lines: only if they match known topic patterns
    topic_patterns = [
        r'(?i)report\s+\(A\d',
        r'(?i)recommendation\s+for',
        r'(?i)proposal\s+for',
        r'(?i)\d+\s+General\s+Budget',
        r'(?i)resolution\s+on',
    ]
    return any(re.search(p, s) for p in topic_patterns)


def format_europarl(text: str) -> str:
    """Transform Europarl raw text to WikiText-style 3-level format."""
    lines = text.splitlines()
    output = []
    topics_found = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 1. Promote speakers: = = Name = = → = = = Name = = =
        m = SPEAKER_RE.match(stripped)
        if m and not L1_RE.match(stripped):
            name = m.group(2)
            output.append(f' = = = {name} = = = ')
            i += 1
            continue

        # 2. Keep level-1 headers as-is
        if L1_RE.match(stripped):
            output.append(line.rstrip())
            i += 1
            continue

        # 3. Check if this is a topic line
        prev = lines[i - 1] if i > 0 else ''
        prev2 = lines[i - 2] if i > 1 else ''
        next_l = lines[i + 1] if i < len(lines) - 1 else ''

        if is_topic_line(line, prev, next_l, prev2):
            # Don't duplicate: check if we already have this as a topic
            output.append(f' = = {stripped} = = ')
            topics_found.append((i + 1, stripped))
            i += 1
            continue

        # 4. Regular line — keep as-is
        output.append(line.rstrip())
        i += 1

    return '\n'.join(output) + '\n', topics_found


def main():
    parser = argparse.ArgumentParser(
        description='Format Europarl raw files to WikiText-style 3-level headers',
    )
    parser.add_argument(
        'input', type=Path,
        help='Input Europarl .raw file',
    )
    parser.add_argument(
        '--output', '-o', type=Path, default=None,
        help='Output file path (default: overwrite input)',
    )
    parser.add_argument(
        '--dry-run', '-n', action='store_true',
        help='Show detected topics without writing',
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f'Error: {args.input} not found')
        return

    text = args.input.read_text(encoding='utf-8')
    result, topics = format_europarl(text)

    # Show summary
    print(f'File: {args.input}')
    print(f'Topics detected: {len(topics)}')
    for lineno, title in topics:
        print(f'  L{lineno:4d}: {title[:80]}')

    # Count speakers
    speaker_count = len(re.findall(r'= = = .+? = = =', result))
    print(f'Speakers promoted: {speaker_count}')

    if args.dry_run:
        print('\n[DRY RUN] No files written.')
        return

    output_path = args.output or args.input
    output_path.write_text(result, encoding='utf-8')
    print(f'\nWritten to: {output_path}')


if __name__ == '__main__':
    main()
