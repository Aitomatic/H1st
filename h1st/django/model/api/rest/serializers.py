from rest_framework.serializers import ModelSerializer, SerializerMethodField

from ...models import Model


class H1stModelSerializer(ModelSerializer):
    description = SerializerMethodField(method_name='get_description')

    class Meta:
        model = Model

        fields = (
            'description',
            'uuid',
            'name',
            'created',
            'modified'
        )

    def get_description(self, obj):
        return str(obj)
