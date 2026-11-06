"""Lets GitHub Actions in one repo use AWS without stored keys (OIDC).

Deploy once from your laptop: cdk deploy GitHubAccess -c repo=<owner>/<repo>
"""
from aws_cdk import CfnOutput, Duration, Stack, aws_iam as iam
from constructs import Construct


class GitHubAccessStack(Stack):
    def __init__(self, scope: Construct, cid: str, repo: str, **kw):
        super().__init__(scope, cid, **kw)
        provider = iam.OpenIdConnectProvider(self, "GitHubOidc",
                                             url="https://token.actions.githubusercontent.com",
                                             client_ids=["sts.amazonaws.com"])
        role = iam.Role(self, "GitHubActionsRole",
                        assumed_by=iam.WebIdentityPrincipal(provider.open_id_connect_provider_arn, conditions={
                            "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
                            "StringLike": {"token.actions.githubusercontent.com:sub": f"repo:{repo}:*"},
                        }),
                        max_session_duration=Duration.hours(1),
                        description=f"GitHub Actions for {repo}")
        # Live runs: call Claude on Bedrock.
        role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=["arn:aws:bedrock:*::foundation-model/*",
                       f"arn:aws:bedrock:*:{self.account}:inference-profile/*"]))
        # Deploys: CDK works by assuming the roles `cdk bootstrap` created.
        role.add_to_policy(iam.PolicyStatement(
            actions=["sts:AssumeRole"], resources=[f"arn:aws:iam::{self.account}:role/cdk-*"]))
        # Publishing reports and reading the stack's outputs.
        role.add_to_policy(iam.PolicyStatement(
            actions=["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
            resources=["arn:aws:s3:::hibobdeploykit-reports*", "arn:aws:s3:::hibobdeploykit-reports*/*"]))
        role.add_to_policy(iam.PolicyStatement(
            actions=["cloudformation:DescribeStacks"], resources=["*"]))
        CfnOutput(self, "RoleArn", value=role.role_arn)
