"""Knowledge Models."""


# pylint: disable=invalid-name


from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    """Knowledge Models."""

    dependencies = [
        ('H1stModel', '0037_PreTrainedHuggingFaceQuestionAnswerer')
    ]

    operations = [
        migrations.CreateModel(
            name='KnowledgeModel',

            fields=[
                ('model_ptr',
                 models.OneToOneField(
                     auto_created=True,
                     on_delete=django.db.models.deletion.CASCADE,
                     parent_link=True,
                     primary_key=True,
                     related_name='h1st_knowledge_models',
                     serialize=False,
                     to='H1stModel.model'))
            ],

            options={
                'verbose_name': 'H1st Knowledge Model',
                'verbose_name_plural': 'H1st Knowledge Models',
                'db_table': 'H1stModel_KnowledgeModel',
                'abstract': False,
                'default_related_name': 'h1st_knowledge_models',
                'base_manager_name': 'objects',
            },

            bases=('H1stModel.model',)
        ),

        migrations.CreateModel(
            name='BooleanLogicKnowledgeModel',

            fields=[
                ('knowledgemodel_ptr',
                 models.OneToOneField(
                     auto_created=True,
                     on_delete=django.db.models.deletion.CASCADE,
                     parent_link=True,
                     primary_key=True,
                     related_name='h1st_boolean_logic_knowledge_models',
                     serialize=False,
                     to='H1stModel.knowledgemodel'))
            ],

            options={
                'verbose_name': 'H1st Boolean Logic Knowledge Model',
                'verbose_name_plural': 'H1st Boolean Logic Knowledge Models',
                'db_table': 'H1stModel_BooleanLogicKnowledgeModel',
                'abstract': False,
                'default_related_name': 'h1st_boolean_logic_knowledge_models',
                'base_manager_name': 'objects',
            },

            bases=('H1stModel.knowledgemodel',)
        )
    ]
