# Generated by Django 3.2.7 on 2021-10-11 06:21


from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [('H1stData', '0007_TextDataSet')]

    operations = [
        migrations.RenameField(
            model_name='jsondataset',
            old_name='json',
            new_name='in_db_json')
    ]
