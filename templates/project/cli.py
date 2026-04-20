from autopilot.cli.main import AutoPilotCLI


class ProjectCLI(AutoPilotCLI, project='{name}'):
  def __init__(self):
    super().__init__()
