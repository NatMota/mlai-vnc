"""Apply shared experiment identity without changing recorded replay data."""
from html import escape
from pathlib import Path
import re

from experiment_labels import ITERATIONS, display_title


def replay_title(results, seed, split):
    path = Path(results)
    name = path.parent.name if path.name in ('val', 'validation', 'train', 'test') else path.name
    title = display_title(name) if name in ITERATIONS else name.replace('_', ' ')
    split = 'validation' if split == 'val' else split
    return f'{title} · seed {seed} · {split}'


def apply_replay_identity(page, results, seed, split):
    title = escape(replay_title(results, seed, split))
    page, titles = re.subn(r'<title>.*?</title>', lambda _: f'<title>{title}</title>', page, count=1, flags=re.S)
    page, headings = re.subn(r'(<div class="top"><h2>).*?(</h2>)', lambda m: m[1] + title + m[2], page, count=1, flags=re.S)
    if titles != 1 or headings != 1:
        raise ValueError('Replay template must contain a document title and top heading')
    return page


if __name__ == '__main__':
    from html import unescape
    template = (Path(__file__).parent.parent / 'driving_30/replay_template.html').read_text()
    titles = []
    for name, iteration in [('baseline', 41), ('inertia', 46), ('random_start_retrained', 49), ('dart', 52)]:
        title = replay_title(Path('/saved') / name / 'test', 7, 'test')
        page = apply_replay_identity(template, Path('/saved') / name / 'test', 7, 'test')
        assert title.startswith(f'Iteration {iteration}: ')
        assert unescape(re.search(r'<title>(.*?)</title>', page)[1]) == title
        assert unescape(re.search(r'<div class="top"><h2>(.*?)</h2>', page)[1]) == title
        assert page.split('<script>', 1)[1] == template.split('<script>', 1)[1].replace(
            'Recorded robot motion · model + ghost expert', escape(title), 1)
        titles.append(title)
    assert len(set(titles)) == 4
    assert replay_title('/saved/dart/val', 7, 'val').endswith('seed 7 · validation')
    print('\n'.join(titles))
