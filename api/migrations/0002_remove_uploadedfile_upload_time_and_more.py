# Generated by Django 5.0.6 on 2024-06-19 06:50

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="uploadedfile",
            name="upload_time",
        ),
        migrations.AddField(
            model_name="uploadedfile",
            name="delimiter",
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
        migrations.AddField(
            model_name="uploadedfile",
            name="model_metadata_path",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name="uploadedfile",
            name="modified_metadata_path",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name="uploadedfile",
            name="selected_delimiter",
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
    ]
