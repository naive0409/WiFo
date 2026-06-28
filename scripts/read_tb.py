from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

path = sys.argv[1]
ea = EventAccumulator(path)
ea.Reload()
tags = ea.Tags()['scalars']
for tag in tags:
    events = ea.Scalars(tag)
    print(f'{tag}:')
    for e in events[::10]:
        print(f'  step={e.step:3d} value={e.value:.4f}')
    print(f'  ... final: step={events[-1].step:3d} value={events[-1].value:.4f}')
    print()
