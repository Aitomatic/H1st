from uuid import UUID

from django.shortcuts import get_object_or_404


# refs:
# - django-rest-framework.org/api-guide/generic-views/#creating-custom-mixins
# - stackoverflow.com/questions/38461366
class LookUpByUUIDorNameMixin:
    def get_object(self):
        # get base queryset
        queryset = self.get_queryset()

        # apply any filter backends
        queryset = self.filter_queryset(queryset)

        # get look-up value
        lookup_value = str(self.kwargs[self.lookup_field])

        try:   # try looking up object by UUID
            _uuid = UUID(lookup_value, version=4)
            obj = get_object_or_404(queryset, uuid=_uuid)

        except ValueError:
            # else look up by Name
            obj = get_object_or_404(queryset, name=lookup_value)

        # check object permissions
        self.check_object_permissions(self.request, obj)

        return obj
