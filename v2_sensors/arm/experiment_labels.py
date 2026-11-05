"""Presentation labels only; experiment keys and evidence paths stay immutable."""
import re


LABELS = {
    'baseline': 'Train arm torque control from reset with physical input scales',
    'encoder_noise': 'Frozen networks: add encoder noise',
    'basics_standard': 'Retrain arm networks with standardised inputs',
    'damping': 'Frozen networks: encoder noise plus damping',
    'elasticity': 'Frozen networks: add elasticity to noise and damping',
    'inertia': 'Frozen networks: add motor inertia to noise, damping and elasticity',
    'delay': 'Frozen networks: add 3 ms observation delay to the cumulative disturbances',
    'random_start_fixed': 'Frozen networks unchanged: random starts with all five disturbances',
    'random_start_retrained': 'Retrain driving and reaching on random starts; evaluate with all five disturbances',
    'random_start_retention': 'Frozen random-start-retrained networks: clean retention at original starts',
    'standard_stack': 'Frozen standardised networks: all five disturbances and random starts',
    'dart': 'DART-inspired recovery training; clean evaluation',
    'dart_stack': 'Frozen DART-trained networks: all five disturbances and random starts',
    'dagger': 'Three-round DAgger recovery training; clean evaluation',
    'dagger_stack': 'Frozen DAgger-trained networks: all five disturbances and random starts',
    'dart_dagger': 'DART-inspired training followed by three-round DAgger; clean evaluation',
    'dart_dagger_stack': 'Frozen DART+DAgger networks: all five disturbances and random starts',
}
ITERATIONS = dict(zip(LABELS, range(41, 58)))


def experiment_label(name):
    """Return displayed iteration and description for an immutable results key."""
    return ITERATIONS[name], LABELS[name]


def display_title(name):
    iteration, label = experiment_label(name)
    return f'Iteration {iteration}: {label}'


def transform_visible_prose(source):
    """Update known Markdown headings, preserving links, paths and provenance.

    Match immutable experiment words, never iteration numbers: old 52–58 have
    already become 51–57. Apply only to visible prose, not notebook code/metadata.
    """
    source = source.replace(' (review and publication pending)', '')
    source = source.replace('(review and publication pending)', '')
    for name in sorted(LABELS, key=len, reverse=True):
        pattern = rf'(?m)^(#+ )Iteration \d+: full-episode {re.escape(name.replace("_", " "))}[ \t]*$'
        source = re.sub(pattern, lambda m: m[1] + display_title(name), source)
    source = source.replace('### Full-episode arm training and cumulative robustness: plan',
                            '### Train arm torque control from reset and evaluate cumulative disturbances: plan')
    source = source.replace('### Overnight comparisons: publication and interpretation',
                            '### Training and frozen-network evaluations: comparison and interpretation')
    return source


if __name__ == '__main__':
    example = ('### Iteration 53: full-episode dart\n\n'
               '(review and publication pending)\n'
               '[evidence](https://host/overnight_20261105/dart/test)\n'
               '`/data/overnight_20261105/dart`')
    result = transform_visible_prose(example)
    assert result.startswith('### Iteration 52: DART-inspired recovery training; clean evaluation')
    assert 'pending' not in result
    assert 'https://host/overnight_20261105/dart/test' in result
    assert '`/data/overnight_20261105/dart`' in result
    assert transform_visible_prose(result) == result
    assert ITERATIONS['random_start_fixed'] == 48
    assert ITERATIONS['random_start_retrained'] == 49
    assert ITERATIONS['random_start_retention'] == 50
    assert ITERATIONS['dart_dagger_stack'] == 57
