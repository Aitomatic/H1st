# Generated by Django 3.2.8 on 2021-10-16 03:55


from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('H1stModel', '0031_PreTrainedHuggingFaceTextSummarizer')
    ]

    operations = [
        migrations.CreateModel(
            name='PreTrainedHuggingFaceTranslator',

            fields=[
                ('pretrainedhuggingfacetransformer_ptr',
                 models.OneToOneField(
                     auto_created=True,
                     on_delete=django.db.models.deletion.CASCADE,
                     parent_link=True,
                     primary_key=True,
                     related_name='h1st_pretrained_hugging_face_translators',
                     serialize=False,
                     to='H1stModel.pretrainedhuggingfacetransformer'))
            ],

            options={
                'verbose_name': 'Pre-Trained Hugging Face Translator',
                'verbose_name_plural': 'Pre-Trained Hugging Face Translators',
                'db_table': 'H1stModel_PreTrainedHuggingFaceTranslator',
                'abstract': False,
                'default_related_name':
                    'h1st_pretrained_hugging_face_translators',
                'base_manager_name': 'objects'
            },

            bases=('H1stModel.pretrainedhuggingfacetransformer',)
        )
    ]
