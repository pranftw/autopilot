"""Agent-optimized prompt: forward -> loss -> backward -> optimizer.step().

The inference agent (no tools) answers questions using prompts/system.txt.
The optimizer agent (file tools) reads text gradients and edits the prompt.
Requires: claude CLI installed.
"""

from autopilot.data.dataloader import DataLoader
from pathlib import Path
from protim.data import QADataset
from protim.module import PromptModule


def example_dir() -> Path:
  return Path(__file__).parent


def main():
  root = example_dir()
  prompts_dir = str(root / 'prompts')
  module = PromptModule(prompts_dir)
  loss = module.loss
  optimizer = module.configure_optimizers()
  metric = module.accuracy

  train_loader = DataLoader(QADataset(root / 'datasets' / 'train.jsonl'))
  prompt_path = Path(prompts_dir) / 'system.txt'

  print('=== Protim: Agent-Optimized Prompt ===\n')
  print(f'System prompt: {prompt_path.read_text().strip()}\n')

  module.train()
  for epoch in range(1, 4):
    metric.reset()
    loss.reset()

    for batch in train_loader:
      data = module(batch)
      loss(data, batch)
      metric.update(data)

    train_metrics = metric.compute()
    print(f'Epoch {epoch}: accuracy={train_metrics["accuracy"]:.2%}')

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f'  Updated prompt: {prompt_path.read_text().strip()[:80]}...\n')

  print('Done.')


if __name__ == '__main__':
  main()
