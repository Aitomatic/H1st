from django.db.models.query_utils import Q
from django.shortcuts import get_object_or_404


# refs:
# - django-rest-framework.org/api-guide/generic-views/#creating-custom-mixins
# - stackoverflow.com/questions/38461366
class LookUpByPKorNameMixin:
    def get_object(self):
        # get base queryset
        queryset = self.get_queryset()

        # apply any filter backends
        queryset = self.filter_queryset(queryset)

        # get look-up value
        lookup_value = self.kwargs['pk']

        # look up object by PK or Name
        obj = get_object_or_404(queryset,
                                Q(pk=lookup_value) | Q(name=lookup_value))

        # check object permissions
        self.check_object_permissions(self.request, obj)

        return obj
