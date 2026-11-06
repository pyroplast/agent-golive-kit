"""Checks on the CloudFormation the CDK stack produces (no AWS account needed)."""
import json

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk.assertions import Match, Template  # noqa: E402

from infra.stack import DeployKitStack  # noqa: E402


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    app = cdk.App(outdir=str(tmp_path_factory.mktemp("cdkout")))
    stack = DeployKitStack(app, "Test", env=cdk.Environment(account="123456789012", region="eu-west-1"))
    return Template.from_stack(stack)


def test_bucket_is_private_and_encrypted(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "PublicAccessBlockConfiguration": {"BlockPublicAcls": True, "BlockPublicPolicy": True,
                                           "IgnorePublicAcls": True, "RestrictPublicBuckets": True},
        "BucketEncryption": Match.any_value(),
    })


def test_report_served_through_cloudfront(template):
    template.has_resource_properties("AWS::CloudFront::Distribution", {
        "DistributionConfig": Match.object_like({"DefaultRootObject": "latest/report.html"}),
    })
    template.resource_count_is("AWS::CloudFront::OriginAccessControl", 1)


def test_nightly_schedule_runs_fargate(template):
    template.has_resource_properties("AWS::Events::Rule", {
        "ScheduleExpression": "cron(17 2 * * ? *)",
        "Targets": Match.array_with([Match.object_like({"EcsParameters": Match.object_like({"LaunchType": "FARGATE"})})]),
    })


def test_task_can_call_bedrock_and_has_bucket_env(template):
    policies = template.find_resources("AWS::IAM::Policy")
    text = json.dumps(policies)
    assert "bedrock:InvokeModel" in text
    td = template.find_resources("AWS::ECS::TaskDefinition")
    env = json.dumps(td)
    assert "REPORT_BUCKET" in env


def test_no_nat_gateway_costs(template):
    template.resource_count_is("AWS::EC2::NatGateway", 0)


def test_github_role_trusts_only_this_repo(tmp_path_factory):
    from infra.github_oidc import GitHubAccessStack
    app = cdk.App(outdir=str(tmp_path_factory.mktemp("gh")))
    stack = GitHubAccessStack(app, "Gh", repo="pyroplast/agent-golive-kit",
                              env=cdk.Environment(account="123456789012", region="eu-west-1"))
    t = Template.from_stack(stack)
    roles = json.dumps(t.find_resources("AWS::IAM::Role"))
    assert "repo:pyroplast/agent-golive-kit:" in roles
    assert "token.actions.githubusercontent.com:aud" in roles
    policies = json.dumps(t.find_resources("AWS::IAM::Policy"))
    assert "bedrock:InvokeModel" in policies and "cdk-" in policies
    t.has_output("RoleArn", {})
