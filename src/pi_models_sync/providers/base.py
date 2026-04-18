from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pi_models_sync.discovery import DiscoveredModel
    from pi_models_sync.providers.config import ProviderConfig


class ModelProvider(abc.ABC):
    """Base interface for a model provider."""

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the provider with a configuration.

        Args:
            config: The configuration for this provider.
        """
        self.config = config

    @abc.abstractmethod
    def get_models(self) -> Iterator[DiscoveredModel]:
        """Fetch and return discovered models.

        Yields:
            DiscoveredModel instances.
        """
        ...
