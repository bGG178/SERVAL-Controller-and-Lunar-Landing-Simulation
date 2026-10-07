"""Repair spacecraft model display metadata in a copy of an existing playback."""
from pathlib import Path
import json
import struct
import sys
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from VizardPlayback import frames, fields, encode_fields, encode_varint
RGBA = b''.join(encode_varint(c) for c in (255, 0, 0, 255))


def repair(source, destination, model_path):
    source, destination, model_path = map(Path, (source, destination, model_path))
    if source.resolve() == destination.resolve():
        raise ValueError('Preserve the original recording; choose a different output')
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    count = changed = 0
    with destination.open('xb') as output:
        for raw, items in frames(source):
            result = []
            modified = False
            for n, wire, value in items:
                if n == 7 and wire == 2:
                    settings = []
                    for k, kw, v in fields(value):
                        if k == 12 and kw == 2:
                            model = list(fields(v))
                            if (2, 2, b'IMX') in model:
                                model = [(j, jw, b) for j, jw, b in model if j not in (1, 5, 9)]
                                model.extend([(1, 2, str(model_path.resolve()).encode()),
                                              (5, 2, struct.pack('<3d', 1., 1., 1.)),
                                              (9, 2, RGBA)])
                                v = encode_fields(model)
                                modified = True
                                changed += 1
                        settings.append((k, kw, v))
                    if modified:
                        value = encode_fields(settings)
                result.append((n, wire, value))
            if modified:
                payload = encode_fields(result)
                output.write(encode_varint(len(payload)) + payload)
            else:
                output.write(raw)
            count += 1
    return dict(frames=count, modified_model_records=changed,
                source=str(source.resolve()), output=str(destination.resolve()), model_path=str(model_path.resolve()))


def verify(source, destination):
    from itertools import zip_longest
    count = 0
    for original, repaired in zip_longest(frames(source), frames(destination)):
        if original is None or repaired is None:
            raise AssertionError('Playback frame count changed')
        # Every timestamp, physical state, actuator packet and ephemeris remains
        # byte-for-byte identical. Only spacecraft display metadata may differ.
        a, b = original[1], repaired[1]
        assert [item for item in a if item[0] != 7] == [item for item in b if item[0] != 7]
        for settings_a, settings_b in zip([v for n, _, v in a if n == 7], [v for n, _, v in b if n == 7]):
            sa, sb = list(fields(settings_a)), list(fields(settings_b))
            assert [item for item in sa if item[0] != 12] == [item for item in sb if item[0] != 12]
            ma, mb = [v for n, _, v in sa if n == 12], [v for n, _, v in sb if n == 12]
            assert len(ma) == len(mb)
            for old_model, new_model in zip(ma, mb):
                old, new = list(fields(old_model)), list(fields(new_model))
                if (2, 2, b'IMX') in old:
                    assert [item for item in old if item[0] not in (1, 5, 9)] == [item for item in new if item[0] not in (1, 5, 9)]
                    assert Path(next(v.decode() for n, _, v in new if n == 1)).is_file()
                    assert next(v for n, _, v in new if n == 5) == struct.pack('<3d', 1., 1., 1.)
                    assert next(v for n, _, v in new if n == 9) == RGBA
                else:
                    assert old_model == new_model
        count += 1
    return count


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    source = root / '_VizFiles' / 'Visualization_UnityViz.bin'
    output = root / '_VizFiles' / 'Visualization_ModelFixed.bin'
    temporary = output.with_name(f'.{output.stem}.{uuid4().hex}.tmp')
    report = repair(source, temporary, root / 'Spacecraft' / 'IM1.obj')
    report['verified_frames'] = verify(source, temporary)
    temporary.replace(output)
    report['output'] = str(output)
    (root / 'work' / 'spacecraft_model_repair_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
