# Generated by Django 3.2.8 on 2021-10-15 07:47


from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('H1stModel', '0027_PreTrainedHuggingFaceTokenClassifier')
    ]

    operations = [
        migrations.CreateModel(
            name='PreTrainedHuggingFaceObjectDetector',

            fields=[
                ('pretrainedhuggingfacetransformer_ptr',
                 models.OneToOneField(
                     auto_created=True,
                     on_delete=django.db.models.deletion.CASCADE,
                     parent_link=True,
                     primary_key=True,
                     related_name=('h1st_pretrained_'
                                   'hugging_face_object_detectors'),
                     serialize=False,
                     to='H1stModel.pretrainedhuggingfacetransformer'))
            ],

            options={
                'verbose_name': 'Pre-Trained Hugging Face Object Detector',
                'verbose_name_plural':
                    'Pre-Trained Hugging Face Object Detectors',
                'db_table': 'H1stModel_PreTrainedHuggingFaceObjectDetector',
                'abstract': False,
                'default_related_name':
                    'h1st_pretrained_hugging_face_object_detectors',
                'base_manager_name': 'objects'
            },

            bases=('H1stModel.pretrainedhuggingfacetransformer',)
        )
    ]
