"""Humanoid agent class."""

from simworld.agent.base_agent import BaseAgent
from simworld.communicator.communicator import Communicator
from simworld.config import Config
from simworld.map.map import Map
from simworld.utils.vector import Vector


class Humanoid(BaseAgent):
    """Humanoid agent class."""

    _id_counter = 0
    _camera_id_counter = 0  # Independent counter for camera IDs

    def __init__(self, position: Vector, direction: Vector, map: Map = None, communicator: Communicator = None, config: Config = None):
        """Initialize humanoid agent.

        Args:
            position: Initial position.
            direction: Initial direction.
            map: Map.
            communicator: Communicator.
            config: Config.
        """
        super().__init__(position, direction)
        self.id = Humanoid._id_counter
        Humanoid._id_counter += 1
        self.camera_id = Humanoid._camera_id_counter
        Humanoid._camera_id_counter += 1

        self.map = map
        self.communicator = communicator
        self.config = config

        # If a communicator is provided, validate that the assigned camera_id exists in UE.
        # If not valid, attempt to find the camera bound to this actor and update camera_id.
        try:
            if self.communicator is not None:
                ue_name = self.communicator.get_humanoid_name(self.id)
                found_cam = self.communicator.find_camera_id_for_actor(ue_name)
                if found_cam is not None:
                    self.camera_id = found_cam
        except Exception:
            # non-fatal — keep assigned camera_id
            pass

        self.scooter_id = None

    def __str__(self):
        """Return a string representation of the humanoid agent."""
        return f'Humanoid(id={self.id}, position={self.position}, direction={self.direction}, scooter_id={self.scooter_id})'

    def __repr__(self):
        """Return a detailed string representation of the humanoid agent."""
        return self.__str__()
