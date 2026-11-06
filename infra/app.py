import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aws_cdk as cdk  # noqa: E402

from infra.github_oidc import GitHubAccessStack  # noqa: E402
from infra.stack import DeployKitStack  # noqa: E402

app = cdk.App()
env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                      region=os.environ.get("CDK_DEFAULT_REGION", "eu-west-1"))
DeployKitStack(app, "HiBobDeployKit", env=env)
GitHubAccessStack(app, "GitHubAccess", env=env,
                  repo=app.node.try_get_context("repo") or "pyroplast/agent-golive-kit")
app.synth()
