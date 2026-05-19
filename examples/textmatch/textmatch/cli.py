from autopilot.cli.main import AutoPilotCLI
from pathlib import Path
from textmatch.data import TextMatchDataModule
from textmatch.module import TextMatchModule


class TextMatchCLI(AutoPilotCLI, project='textmatch'):
  def __init__(self):
    super().__init__()
    root = Path(__file__).parent.parent
    self.module = TextMatchModule(str(root / 'rules'))
    self.datamodule = TextMatchDataModule(str(root / 'datasets'))
