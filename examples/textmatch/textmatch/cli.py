from autopilot.cli.main import AutoPilotCLI


class TextMatchCLI(AutoPilotCLI, project='textmatch'):
  def __init__(self):
    super().__init__()
