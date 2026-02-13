"""This module provides functionality for retrieving and placing assets in the city simulation."""
import importlib.resources as pkg_resources
import os

from sentence_transformers import SentenceTransformer

from simworld.assets_rp.utils.assets_rp_utils import (
    construct_building_from_candidate, get_coordinates_around_building,
    get_parsed_input, get_surroundings, place_target_asset,
    retrieve_target_asset, vector_cosine_similarity)
from simworld.assets_rp.utils.reference_assets_retriever import \
    ReferenceAssetsRetriever
from simworld.citygen.dataclass.dataclass import Point
from simworld.config import Config
from simworld.utils.data_importer import DataImporter
from simworld.utils.logger import Logger



from simworld.communicator.communicator import Communicator


class AssetsRetrieverPlacer:
    """Assets Retrieval and Placement ability.

    This class provides methods to retrieve the assets and place them somewhere
    based on the natural language prompts.
    """

    def __init__(self, config: Config, input_dir: str, env_description_retrieval_model_name: str = None, communicator: Communicator = None):
        """Initialize function call.

        Args:
            config: the configuration user provide.
            input_dir: the directory to load the input data.
            env_description_retrieval_model_name: the name of the environment description retrieval model.
            communicator: Optional Communicator instance for real-time asset placement.
        """
        self.config = config
        self.env_description_retrieval_model_name = env_description_retrieval_model_name if env_description_retrieval_model_name else config['assets_rp.env_description_retrieval_model']
        self.model = SentenceTransformer(self.env_description_retrieval_model_name)
        self.data_importer = DataImporter(config)
        self.input_dir = input_dir
        self.city_generator = self.data_importer.import_city_data(input_dir)
        self.communicator = communicator

        self.logger = Logger.get_logger('AssetsRP')

    def generate_assets_manually(self, natural_language_input, sample_dataset_dir: str = None, output_dir: str = None,  description_map_path: str = None, assets_retrieval_model: str = None):
        """This function is used to retrieve and place the assets based on user's prompt.

        Args:
            natural_language_input: the text prompt provided by the users.
            sample_dataset_dir: the directory to load the images of the assets.
            output_dir: the directory to save the output.
            description_map_path: the path to the description map file.
            assets_retrieval_model: the name of the assets retrieval model.
        """
        # 1. Parse the input
        parsed_input, asset_to_place, reference_asset_query, relation, surroundings_query = get_parsed_input(natural_language_input)

        self.logger.info('LLM parse result: %s', parsed_input)

        # 2. Load the file that store all the assets. Find the candidates that match "reference_asset_query"
        _progen_world_path = os.path.join(self.input_dir, 'progen_world.json')
        _description_map_path = description_map_path if description_map_path else self.config['assets_rp.input_description_map']
        referenceAssetRetriever = ReferenceAssetsRetriever(_progen_world_path, _description_map_path, self.env_description_retrieval_model_name)
        candidate_nodes = referenceAssetRetriever.retrieve_reference_assets(reference_asset_query)

        # 3. For each candidate, use "_get_point_around_label" to obtain its surrounding asset
        candidate_similarity_scores = []
        for candidate, base_score in candidate_nodes:
            x = candidate['properties']['location']['x'] / 100
            y = candidate['properties']['location']['y'] / 100
            node_position = Point(x, y)
            candidate_surroundings = self.city_generator.route_generator.get_point_around_label(node_position, self.city_generator.city_quadtrees, 200, 20)

            # 4. Integrate the surrounding information to a string and do embedding, and calculate the similarity score.
            candidate_surroundings_str = get_surroundings(candidate_surroundings, _description_map_path)
            candidate_embedding = self.model.encode(candidate_surroundings_str)
            query_embedding = self.model.encode(surroundings_query)

            similarity = vector_cosine_similarity(candidate_embedding, query_embedding)
            candidate_similarity_scores.append((candidate, similarity))
            # self.logger.info(f"candidate nodes: {candidate['id']} similarity score: {similarity:.4f}")

        # 5. Choose the highest score as final reference asset and construct the instance
        best_candidate, best_similarity = max(candidate_similarity_scores, key=lambda x: x[1])
        self.logger.info('best candidate: %s similarity score: %s', best_candidate['id'], best_similarity)

        reference_asset = construct_building_from_candidate(best_candidate, os.path.join(self.input_dir, 'buildings.json'))
        if reference_asset is None:
            self.logger.error('No reference asset found for %s', best_candidate['id'])
            return

        # 6. Use CLIP to obtain the asset and place around the best candidate
        if sample_dataset_dir is None:
            self.logger.info('No sample dataset directory provided, using default')
            _sample_dataset_dir = pkg_resources.files('simworld.data').joinpath(self.config['assets_rp.input_sample_dataset'])
        else:
            _sample_dataset_dir = sample_dataset_dir

        if assets_retrieval_model is None:
            self.logger.info('No assets retrieval model provided, using default')
            _assets_retrieval_model = self.config['assets_rp.assets_retrieval_model']
        else:
            _assets_retrieval_model = assets_retrieval_model

        self.logger.info('Using %s to retrieve target assets', _assets_retrieval_model)
        target_assets = retrieve_target_asset(asset_to_place, _sample_dataset_dir, _assets_retrieval_model)
        self.logger.info('target assets: %s', target_assets)
        target_positions = get_coordinates_around_building(self.city_generator.config, reference_asset, relation, len(target_assets))
        self.logger.info('target positions: %s', target_positions)
        if len(target_assets) == 0 or len(target_positions) == 0:
            self.logger.error('No target assets or target positions found')
            return
        place_target_asset(target_assets, target_positions, output_dir or self.config['assets_rp.output_dir'])


    def place_asset(self, asset_path: str, location: tuple, rotation: tuple = (0, 0, 0), scale: tuple = (1, 1, 1)):
        """Directly place an asset in the simulation using the provided Communicator.
        
        Args:
            asset_path: The full path to the blueprint asset (e.g., /Game/ModularSciFI/SM_Apple/SM_Apple.SM_Apple)
            location: Tuple of (x, y, z) coordinates.
            rotation: Tuple of (pitch, yaw, roll) rotation values. Defaults to (0, 0, 0).
            scale: Tuple of (x, y, z) scale values. Defaults to (1, 1, 1).
        """
        if not self.communicator:
            self.logger.error("Cannot place asset: No Communicator instance provided.")
            return False

        # Generate a unique name for the asset
        import time
        timestamp = int(time.time() * 1000)
        # Extract a simple name from the path (e.g. SM_Apple)
        base_name = asset_path.split('/')[-1].split('.')[0]
        object_name = f'BP_Direct_{base_name}_{timestamp}'
        
        self.logger.info(f"Attempting to spawn asset '{object_name}' from path '{asset_path}'")
        self.logger.info(f"Target Location: {location}, Rotation: {rotation}, Scale: {scale}")
        
        try:
            # 1. Spawn
            self.logger.info(f"Sending spawn command for {asset_path}...")
            # Note: Communicator's spawn_bp_asset is a wrapper around client.request('vset /objects/spawn_bp_asset ...')
            # It strictly requires the asset path to be valid in the cooked project.
            self.communicator.unrealcv.spawn_bp_asset(asset_path, object_name)
            
            # Verify if spawn command was actually successful by checking if object exists
            # vget /objects returns a list of all objects. This might be heavy if there are many objects.
            # Alternatively, try to get location of the new object. If it fails, spawn likely failed.
            
            check_response = self.communicator.unrealcv.client.request(f'vget /object/{object_name}/location')
            if "error" in check_response.lower() or "null" in check_response.lower():
                 self.logger.error(f"Spawn possibly failed. UnrealCV response to location check: {check_response}")
                 self.logger.error(f"Please verify that the asset path '{asset_path}' exists in the cooked game content.")
                 return False

            # 2. Set Location 
            self.logger.info(f"Setting location to {location}...")
            self.communicator.unrealcv.set_location(location, object_name)
            
            # 3. Set Orientation
            self.logger.info(f"Setting orientation to {rotation}...")
            self.communicator.unrealcv.set_orientation(rotation, object_name)
            
            # 4. Set Scale
            self.logger.info(f"Setting scale to {scale}...")
            self.communicator.unrealcv.set_scale(scale, object_name)
            
            # 5. Set properties
            self.communicator.unrealcv.set_collision(object_name, True)
            self.communicator.unrealcv.set_movable(object_name, True)
            
            self.logger.info(f"Successfully placed {object_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to place asset '{object_name}': {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


