import os
import xml.etree.ElementTree as ET
from threading import Lock
from pathlib import Path


class DataRepo:
    _instance = None
    _lock = Lock()

    BASE_DIR = Path(__file__).parent.resolve()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.data_dir = self.BASE_DIR / "data"
        self.config_dir = self.BASE_DIR / "config"
        self.cctv_dir = self.BASE_DIR / "cctv_data"
        self._files = {}
        self._xml_cache = {}
        self._scan()

    def _scan(self):
        for directory in (self.data_dir, self.config_dir, self.cctv_dir):
            if not directory.exists():
                continue
            for root, _dirs, files in os.walk(directory):
                for fname in files:
                    fpath = Path(root) / fname
                    rel = fpath.relative_to(self.BASE_DIR)
                    key = str(rel).replace("\\", "/")
                    self._files[key] = fpath

    @property
    def sumo_dir(self):
        return self.data_dir / "sumo_files"

    @property
    def sumo_config_path(self):
        return self._files.get("data/sumo_files/map_suhat_sumoconfig.sumocfg")

    @property
    def net_path(self):
        return self._files.get("data/sumo_files/map_suhat_edit.net.xml")

    @property
    def route_path(self):
        return self._files.get("data/sumo_files/map_suhat_netedit.rou.xml")

    @property
    def induction_loop_path(self):
        return self._files.get("data/sumo_files/induction_loop.xml")

    @property
    def all_files(self):
        return dict(self._files)

    @property
    def xml_files(self):
        return {k: v for k, v in self._files.items() if v.suffix.lower() in (".xml", ".sumocfg", ".rou.xml", ".net.xml")}

    @property
    def osm_files(self):
        return {k: v for k, v in self._files.items() if v.suffix.lower() == ".osm"}

    def get_path(self, key):
        return self._files.get(key)

    def read_text(self, key):
        path = self._files.get(key)
        if path:
            return path.read_text(encoding="utf-8")

    def read_xml(self, key):
        if key in self._xml_cache:
            return self._xml_cache[key]
        path = self._files.get(key)
        if path and path.suffix.lower() in (".xml", ".sumocfg", ".rou.xml", ".net.xml"):
            tree = ET.parse(path)
            self._xml_cache[key] = tree
            return tree

    def get_sumo_config(self):
        tree = self.read_xml("data/sumo_files/map_suhat_sumoconfig.sumocfg")
        if tree is None:
            return {}
        root = tree.getroot()
        ns = {"ns": "http://sumo.dlr.de/xsd/sumoConfiguration.xsd"}
        config = {}
        for elem in root.iter():
            if elem.tag == "net-file":
                config["net_file"] = elem.get("value")
            elif elem.tag == "route-files":
                config["route_files"] = elem.get("value")
            elif elem.tag == "additional-files":
                config["additional_files"] = elem.get("value")
        return config

    def get_routes(self):
        tree = self.read_xml("data/sumo_files/map_suhat_netedit.rou.xml")
        if tree is None:
            return []
        routes = []
        for flow in tree.iter("flow"):
            routes.append({
                "id": flow.get("id"),
                "begin": float(flow.get("begin", 0)),
                "from": flow.get("from"),
                "to": flow.get("to"),
                "end": float(flow.get("end", 0)),
                "vehs_per_hour": float(flow.get("vehsPerHour", 0)),
            })
        return routes

    def get_induction_loops(self):
        tree = self.read_xml("data/sumo_files/induction_loop.xml")
        if tree is None:
            return []
        loops = []
        for loop in tree.iter("inductionLoop"):
            loops.append({
                "id": loop.get("id"),
                "lane": loop.get("lane"),
                "pos": float(loop.get("pos", 0)),
                "freq": int(loop.get("freq", 0)),
                "file": loop.get("file"),
            })
        return loops

    def get_edges(self):
        tree = self.read_xml("data/sumo_files/map_suhat_edit.net.xml")
        if tree is None:
            return []
        edges = []
        root = tree.getroot()
        for edge in root.iter("edge"):
            if edge.get("function") == "internal":
                continue
            lanes = []
            for lane in edge.iter("lane"):
                lanes.append({
                    "id": lane.get("id"),
                    "index": int(lane.get("index", 0)),
                    "speed": float(lane.get("speed", 0)),
                    "length": float(lane.get("length", 0)),
                })
            edges.append({
                "id": edge.get("id"),
                "from": edge.get("from"),
                "to": edge.get("to"),
                "priority": edge.get("priority"),
                "type": edge.get("type"),
                "lanes": lanes,
            })
        return edges

    def __repr__(self):
        return f"DataRepo(files={len(self._files)})"
