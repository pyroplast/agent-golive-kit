"""AWS stack: private S3 bucket + CloudFront for reports, nightly Fargate regression run on Bedrock."""
from pathlib import Path

from aws_cdk import (CfnOutput, Duration, RemovalPolicy, Stack, aws_cloudfront as cf,
                     aws_cloudfront_origins as origins, aws_ec2 as ec2, aws_ecs as ecs,
                     aws_events as events, aws_events_targets as targets, aws_iam as iam,
                     aws_logs as logs, aws_s3 as s3)
from constructs import Construct

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DeployKitStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kw):
        super().__init__(scope, cid, **kw)

        bucket = s3.Bucket(self, "Reports",
                           block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                           encryption=s3.BucketEncryption.S3_MANAGED, enforce_ssl=True,
                           removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        dist = cf.Distribution(self, "ReportSite",
                               default_root_object="latest/report.html",
                               default_behavior=cf.BehaviorOptions(
                                   origin=origins.S3BucketOrigin.with_origin_access_control(bucket),
                                   viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS))

        # Public subnets only, no NAT gateway: the task gets a public IP to reach AWS APIs.
        vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=0,
                      subnet_configuration=[ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC)])
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        task = ecs.FargateTaskDefinition(self, "Task", cpu=512, memory_limit_mib=1024)
        task.add_container("deploykit",
                           image=ecs.ContainerImage.from_asset(
                               str(PROJECT_ROOT), file="Dockerfile",
                               exclude=["runs", "cdk.out", ".git", "tests", "**/__pycache__", ".venv"]),
                           environment={"REPORT_BUCKET": bucket.bucket_name, "AWS_REGION": self.region},
                           logging=ecs.LogDrivers.aws_logs(stream_prefix="deploykit",
                                                           log_retention=logs.RetentionDays.ONE_MONTH))
        bucket.grant_read_write(task.task_role)
        task.task_role.add_to_principal_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=["arn:aws:bedrock:*::foundation-model/*",
                       f"arn:aws:bedrock:*:{self.account}:inference-profile/*"]))

        events.Rule(self, "Nightly",
                    schedule=events.Schedule.cron(minute="17", hour="2"),
                    targets=[targets.EcsTask(cluster=cluster, task_definition=task,
                                             subnet_selection=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
                                             assign_public_ip=True)])

        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "ReportUrl", value=f"https://{dist.distribution_domain_name}/latest/report.html")
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "TaskDefinition", value=task.task_definition_arn)
