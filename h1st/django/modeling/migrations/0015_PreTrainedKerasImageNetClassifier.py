# Generated by Django 3.2.7 on 2021-10-01 01:01


from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('H1stModel',
         '0014_PyLoadablePreTrainedMLModel_py_loader_module_and_qualname')
    ]

    operations = [
        migrations.CreateModel(
            name='PreTrainedKerasImageNetClassifier',

            fields=[
                ('pyloadablepretrainedmlmodel_ptr',
                 models.OneToOneField(
                     auto_created=True,
                     on_delete=django.db.models.deletion.CASCADE,
                     parent_link=True,
                     primary_key=True,
                     serialize=False,
                     to='H1stModel.pyloadablepretrainedmlmodel')),

                ('preprocessor_module_and_qualname',
                 models.CharField(
                     db_index=True,
                     default=None,
                     help_text='Preprocessor (module.qualname)',
                     max_length=255,
                     verbose_name='Preprocessor (module.qualname)'))
            ],

            options={
                'verbose_name': 'Pre-Trained Keras ImageNet Classifier',
                'verbose_name_plural':
                    'Pre-Trained Keras ImageNet Classifiers',
            },

            bases=('H1stModel.pyloadablepretrainedmlmodel',)
        )
    ]
