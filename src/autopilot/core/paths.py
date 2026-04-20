from pathlib import Path


def autopilot_dir(workspace: Path) -> Path:
  return workspace / 'autopilot'


def root(workspace: Path, project: str | None = None) -> Path:
  """Base for project-scoped paths. Falls back to autopilot_dir when no project."""
  return projects_dir(workspace) / project if project else autopilot_dir(workspace)


def experiments(workspace: Path, project: str | None = None) -> Path:
  return root(workspace, project) / 'experiments'


def experiment(workspace: Path, slug: str, project: str | None = None) -> Path:
  return experiments(workspace, project) / slug


def datasets(workspace: Path, project: str | None = None) -> Path:
  return root(workspace, project) / 'datasets'


def records(workspace: Path, project: str | None = None) -> Path:
  return root(workspace, project) / 'records'


def project_cli(workspace: Path, project: str) -> Path:
  return root(workspace, project) / 'cli.py'


def dataset_split(workspace: Path, split: str, filename: str, project: str | None = None) -> Path:
  return datasets(workspace, project) / split / filename


def projects_dir(workspace: Path) -> Path:
  return autopilot_dir(workspace) / 'projects'


def split_summary(experiment_dir: Path, split: str) -> Path:
  return experiment_dir / f'{split}_summary.json'


def manifest_file(experiment_dir: Path) -> Path:
  return experiment_dir / 'manifest.json'


def events_file(experiment_dir: Path) -> Path:
  return experiment_dir / 'events.jsonl'


def commands_file(experiment_dir: Path) -> Path:
  return experiment_dir / 'commands.json'


def result_file(experiment_dir: Path) -> Path:
  return experiment_dir / 'result.json'


def promotion_file(experiment_dir: Path) -> Path:
  return experiment_dir / 'promotion.json'


def store(workspace: Path, project: str | None = None) -> Path:
  return root(workspace, project) / '.store'


def epoch_dir(experiment_dir: Path, epoch: int) -> Path:
  return experiment_dir / f'epoch_{epoch}'


def epoch_artifact(experiment_dir: Path, epoch: int, filename: str) -> Path:
  return epoch_dir(experiment_dir, epoch) / filename


def best_baseline_file(experiment_dir: Path) -> Path:
  return experiment_dir / 'best_baseline.json'


def hypothesis_log_file(experiment_dir: Path) -> Path:
  return experiment_dir / 'hypothesis_log.jsonl'


def verdict_file(experiment_dir: Path, epoch: int) -> Path:
  return epoch_dir(experiment_dir, epoch) / 'proposal_verdict.json'


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[3]


def templates_dir() -> Path:
  return _repo_root() / 'templates'


def project_templates_dir() -> Path:
  return templates_dir() / 'project'
