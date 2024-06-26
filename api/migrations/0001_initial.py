# Generated by Django 5.0.6 on 2024-06-18 10:12

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="UploadedFile",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("file", models.FileField(upload_to="uploads/")),
                ("original_name", models.CharField(max_length=255)),
                ("upload_time", models.DateTimeField(auto_now_add=True)),
                (
                    "profile_report_path",
                    models.CharField(blank=True, max_length=255, null=True),
                ),
                ("model_path", models.CharField(blank=True, max_length=255, null=True)),
                ("profile_generated", models.BooleanField(default=False)),
                ("model_trained", models.BooleanField(default=False)),
            ],
        ),
    ]
