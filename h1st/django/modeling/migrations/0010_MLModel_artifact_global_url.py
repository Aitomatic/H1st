# Generated by Django 3.2.7 on 2021-09-30 07:15


from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('H1stModel', '0009_MLModel')]

    operations = [
        migrations.AddField(
            model_name='mlmodel',
            name='artifact_global_url',
            field=models.CharField(
                blank=True,
                db_index=True,
                default=None,
                help_text='ML Model Artifact Global URL',
                max_length=255,
                null=True,
                unique=True,
                verbose_name='ML Model Artifact Global URL'))
    ]
