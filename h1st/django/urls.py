from collections.abc import Sequence

from django.urls.conf import include, path
from django.urls.resolvers import URLPattern

from .data_mgmt import urls as data_mgmt_urls
from .modeling import urls as modeling_urls
from .trust_vault import urls as trust_vault_urls


__all__: Sequence[str] = ('urlpatterns',)


urlpatterns: Sequence[URLPattern] = (
    path(route='data/',
         view=include(data_mgmt_urls)),

    path(route='modeling/',
         view=include(modeling_urls)),
    path(route='models/',
         view=include(modeling_urls)),
    path(route='model/',
         view=include(modeling_urls)),

    # TODO: enhance
    path(route='workflows/',
         view=include(modeling_urls)),
    path(route='workflow/',
         view=include(modeling_urls)),
    path(route='graphs/',
         view=include(modeling_urls)),
    path(route='graph/',
         view=include(modeling_urls)),

    path(route='trust/',
         view=include(trust_vault_urls)),
)
