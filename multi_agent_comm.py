import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from config import Config
from data_processor import DataProcessor
from semantic_encoder import SemanticEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentCommunicator:
    """
    Multi-Agent Communicator class for enhanced AI project based on research paper.

    This class implements the Cooperative Semantic Communication (Co-SC) framework
    for multi-vehicle communication. It involves semantic encoders at each vehicle
    that extract and transmit task-relevant semantics of the data.

    ...

    Attributes
    ----------
    config : Config
        Configuration object containing project settings.

    data_processor : DataProcessor
        Data processing module to handle data preprocessing and normalization.

    semantic_encoder : SemanticEncoder
        Semantic encoder model to extract semantic information from raw data.

    vehicles : Dict[str, VehicleNode]
        Dictionary to store VehicleNode objects with vehicle IDs as keys.

    data : List[Dict]
        List of dictionaries containing raw data from each vehicle.

    semantic_data : List[Tensor]
        List of semantic tensors encoded from raw data.

    transmission_history : List[Dict]
        History of transmitted data between vehicles.

    Methods
    -------
    process_data(self, vehicle_id: str, data: Dict) -> None:
        Processes and validates raw data from a vehicle.

    encode_semantics(self, data: Dict) -> Tensor:
        Encodes semantic information from processed data.

    transmit_data(self, sending_vehicle: str, receiving_vehicles: List[str]) -> None:
        Transmits semantic data from one vehicle to multiple receiving vehicles.

    receive_data(self, receiving_vehicle: str) -> Optional[Tensor]:
        Receives and returns semantic data for a specific vehicle.

    update_vehicle_data(self, vehicle_id: str, data: Dict) -> None:
        Updates the raw data of a specific vehicle and triggers data processing.

    start(self) -> None:
        Starts the multi-agent communication process.

    stop(self) -> None:
        Stops the multi-agent communication process and performs cleanup.

    """

    def __init__(self, config: Config):
        """
        Initializes the MultiAgentCommunicator with required modules and configurations.

        Parameters
        ----------
        config : Config
            Configuration object containing project settings.

        """
        self.config = config
        self.data_processor = DataProcessor(config)
        self.semantic_encoder = SemanticEncoder(config)
        self.vehicles = {}
        self.data = []
        self.semantic_data = []
        self.transmission_history = []
        self.lock = threading.Lock()

    def process_data(self, vehicle_id: str, data: Dict) -> None:
        """
        Processes and validates raw data from a vehicle.

        Parameters
        ----------
        vehicle_id : str
            Unique identifier of the vehicle sending the data.

        data : Dict
            Raw data captured by the vehicle, containing sensor readings.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the data is missing required fields or fails validation checks.

        """
        try:
            processed_data = self.data_processor.process(data)
            self.data.append(processed_data)
            logger.info(f"Processed data from vehicle {vehicle_id}: {processed_data}")
        except ValueError as e:
            logger.error(f"Error processing data from vehicle {vehicle_id}: {e}")
            raise ValueError(f"Invalid data format or values from vehicle {vehicle_id}.")

    def encode_semantics(self, data: Dict) -> Tensor:
        """
        Encodes semantic information from processed data.

        Parameters
        ----------
        data : Dict
            Processed data from a vehicle, containing normalized sensor readings.

        Returns
        -------
        Tensor
            Semantic tensor representation of the data.

        """
        semantic_tensor = self.semantic_encoder.encode(data)
        return semantic_tensor

    def transmit_data(self, sending_vehicle: str, receiving_vehicles: List[str]) -> None:
        """
        Transmits semantic data from one vehicle to multiple receiving vehicles.

        Parameters
        ----------
        sending_vehicle : str
            Unique identifier of the vehicle sending the data.

        receiving_vehicles : List[str]
            List of unique identifiers of the vehicles that should receive the data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the sending vehicle does not exist or has not sent any data.

        """
        if sending_vehicle not in self.vehicles:
            raise ValueError(f"Sending vehicle {sending_vehicle} does not exist.")

        semantic_data = self.semantic_data[self.vehicles[sending_vehicle].data_index]

        for receiving_vehicle in receiving_vehicles:
            if receiving_vehicle in self.vehicles:
                receiving_vehicle_node = self.vehicles[receiving_vehicle]
                receiving_vehicle_node.receive_data(semantic_data)
                transmission_record = {
                    "sending_vehicle": sending_vehicle,
                    "receiving_vehicle": receiving_vehicle,
                    "data": semantic_data.tolist()
                }
                self.transmission_history.append(transmission_record)
                logger.info(f"Transmitted data from vehicle {sending_vehicle} to vehicle {receiving_vehicle}.")
            else:
                logger.warning(f"Receiving vehicle {receiving_vehicle} does not exist.")

    def receive_data(self, receiving_vehicle: str) -> Optional[Tensor]:
        """
        Receives and returns semantic data for a specific vehicle.

        Parameters
        ----------
        receiving_vehicle : str
            Unique identifier of the vehicle that should receive the data.

        Returns
        -------
        Optional[Tensor]
            Semantic tensor representation of the received data, or None if no data is available.

        """
        if receiving_vehicle in self.vehicles:
            receiving_vehicle_node = self.vehicles[receiving_vehicle]
            return receiving_vehicle_node.get_data()
        else:
            logger.warning(f"Receiving vehicle {receiving_vehicle} does not exist.")
            return None

    def update_vehicle_data(self, vehicle_id: str, data: Dict) -> None:
        """
        Updates the raw data of a specific vehicle and triggers data processing.

        Parameters
        ----------
        vehicle_id : str
            Unique identifier of the vehicle sending the data.

        data : Dict
            Raw data captured by the vehicle, containing sensor readings.

        Returns
        -------
        None

        """
        if vehicle_id in self.vehicles:
            self.vehicles[vehicle_id].update_data(data)
            self.process_data(vehicle_id, data)
        else:
            vehicle_node = VehicleNode(vehicle_id)
            vehicle_node.update_data(data)
            self.vehicles[vehicle_id] = vehicle_node
            self.process_data(vehicle_id, data)

    def start(self) -> None:
        """
        Starts the multi-agent communication process.

        Initializes the data processing and encoding, and starts a thread for each vehicle
        to handle data transmission and reception.

        Returns
        -------
        None

        """
        for vehicle_id, data in self.config.initial_data.items():
            self.update_vehicle_data(vehicle_id, data)

        for vehicle_id in self.vehicles:
            vehicle_node = self.vehicles[vehicle_id]
            data_index = vehicle_node.data_index
            semantic_tensor = self.encode_semantics(self.data[data_index])
            vehicle_node.set_data(semantic_tensor)
            self.semantic_data.append(semantic_tensor)

        for vehicle_id in self.vehicles:
            vehicle_thread = threading.Thread(target=self.transmit_and_receive, args=(vehicle_id,))
            vehicle_thread.start()

    def stop(self) -> None:
        """
        Stops the multi-agent communication process and performs cleanup.

        Joins all vehicle threads and clears data structures.

        Returns
        -------
        None

        """
        for vehicle_thread in threading.enumerate():
            if vehicle_thread.name.startswith("VehicleThread") and vehicle_thread.is_alive():
                vehicle_thread.join()

        self.data.clear()
        self.semantic_data.clear()
        self.transmission_history.clear()
        self.vehicles.clear()
        logger.info("Multi-agent communication process stopped.")

class VehicleNode:
    """
    VehicleNode class to manage data transmission and reception for each vehicle.

    Attributes
    ----------
    vehicle_id : str
        Unique identifier of the vehicle.

    data : Optional[Tensor]
        Semantic tensor representation of the vehicle's data.

    data_index : int
        Index of the vehicle's data in the shared data list.

    Methods
    -------
    update_data(self, data: Dict) -> None:
        Updates the raw data of the vehicle.

    receive_data(self, data: Tensor) -> None:
        Receives and stores semantic data for the vehicle.

    get_data(self) -> Optional[Tensor]:
        Returns the semantic data of the vehicle.

    set_data(self, data: Tensor) -> None:
        Sets the semantic data of the vehicle.

    """

    def __init__(self, vehicle_id: str):
        """
        Initializes the VehicleNode with a unique vehicle identifier.

        Parameters
        ----------
        vehicle_id : str
            Unique identifier of the vehicle.

        """
        self.vehicle_id = vehicle_id
        self.data: Optional[Tensor] = None
        self.data_index: int = -1

    def update_data(self, data: Dict) -> None:
        """
        Updates the raw data of the vehicle.

        Parameters
        ----------
        data : Dict
            Raw data captured by the vehicle, containing sensor readings.

        Returns
        -------
        None

        """
        self.data = None

    def receive_data(self, data: Tensor) -> None:
        """
        Receives and stores semantic data for the vehicle.

        Parameters
        ----------
        data : Tensor
            Semantic tensor representation of the received data.

        Returns
        -------
        None

        """
        self.data = data

    def get_data(self) -> Optional[Tensor]:
        """
        Returns the semantic data of the vehicle.

        Returns
        -------
        Optional[Tensor]
            Semantic tensor representation of the vehicle's data, or None if no data is available.

        """
        return self.data

    def set_data(self, data: Tensor) -> None:
        """
        Sets the semantic data of the vehicle.

        Parameters
        ----------
        data : Tensor
            Semantic tensor representation of the vehicle's data.

        Returns
        -------
        None

        """
        self.data = data

def main():
    config = Config("config.yaml")
    communicator = MultiAgentCommunicator(config)
    communicator.start()
    input("Press Enter to stop the multi-agent communication process...")
    communicator.stop()

if __name__ == "__main__":
    main()