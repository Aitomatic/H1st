# Generated by Django 3.2.7 on 2021-10-05 07:14


from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [('H1stModel', '0017_CloudServiceModel')]

    operations = [
        migrations.CreateModel(
            name='GoogleCloudTranslationServiceModel',

            fields=[
                ('cloudservicemodel_ptr',
                 models.OneToOneField(
                     auto_created=True,
                     on_delete=django.db.models.deletion.CASCADE,
                     parent_link=True,
                     primary_key=True,
                     related_name=
                     'h1st_google_cloud_translation_service_models',
                     serialize=False,
                     to='H1stModel.cloudservicemodel'))
            ],

            options={
                'verbose_name': 'Google Cloud Translation Service Model',
                'verbose_name_plural':
                    'Google Cloud Translation Service Models',
                'db_table': 'H1stModel_GoogleCloudTranslationServiceModel',
                'abstract': False,
                'default_related_name':
                    'h1st_google_cloud_translation_service_models',
                'base_manager_name': 'objects'
            },

            bases=('H1stModel.cloudservicemodel',)
        ),

        migrations.CreateModel(
            name='GoogleTranslateServiceModel',

            fields=[
                ('cloudservicemodel_ptr',
                 models.OneToOneField(
                     auto_created=True,
                     on_delete=django.db.models.deletion.CASCADE,
                     parent_link=True,
                     primary_key=True,
                     related_name='h1st_google_translate_service_models',
                     serialize=False,
                     to='H1stModel.cloudservicemodel'))
            ],

            options={
                'verbose_name': 'Google Translate Service Model',
                'verbose_name_plural': 'Google Translate Service Models',
                'db_table': 'H1stModel_GoogleTranslateServiceModel',
                'abstract': False,
                'default_related_name': 'h1st_google_translate_service_models',
                'base_manager_name': 'objects'
            },

            bases=('H1stModel.cloudservicemodel',)
        )
    ]
