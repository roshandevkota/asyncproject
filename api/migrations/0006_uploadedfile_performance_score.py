# Generated by Django 5.0.6 on 2024-06-19 22:57

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0005_uploadedfile_model_type_uploadedfile_target_type_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="uploadedfile",
            name="performance_score",
            field=models.FloatField(blank=True, null=True),
        ),
    ]
