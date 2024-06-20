from django.db import models
import uuid

class UploadedFile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(upload_to='uploads/')
    original_name = models.CharField(max_length=255)
    profile_generated = models.BooleanField(default=False)
    profile_report_path = models.CharField(max_length=255, blank=True, null=True)
    model_trained = models.BooleanField(default=False)
    model_metadata_path = models.CharField(max_length=255, blank=True, null=True)
    modified_metadata_path = models.CharField(max_length=255, blank=True, null=True)
    model_path = models.CharField(max_length=255, blank=True, null=True)
    delimiter = models.CharField(max_length=10, blank=True, null=True)
    selected_delimiter = models.CharField(max_length=10, blank=True, null=True)
    target_column = models.CharField(max_length=255, blank=True, null=True)
    num_trials = models.IntegerField(blank=True, null=True)
    train_score = models.FloatField(blank=True, null=True)
    target_type = models.CharField(max_length=50, blank=True, null=True)
    model_type = models.CharField(max_length=50, blank=True, null=True)
    performance_score = models.FloatField(blank=True, null=True)
    

    def __str__(self):
        return self.original_name
