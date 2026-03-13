import folium
import math
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
import pandas as pd


ROME_HOSPITALS = [
    # === CENTRO ROMA ===
    ("Policlinico Umberto I", 41.9054, 12.5109),
    ("Policlinico Gemelli", 41.9319, 12.4279),
    ("Ospedale San Camillo", 41.8697, 12.4654),
    ("Ospedale Fatebenefratelli Isola Tiberina", 41.8910, 12.4775),
    ("Ospedale Sant'Andrea", 41.9512, 12.4918),
    ("Ospedale Pertini", 41.9189, 12.5561),
    ("Ospedale Sant'Eugenio", 41.8317, 12.4908),
    ("Istituto Regina Elena IFO", 41.8256, 12.4253),
    #("Centro Ospedaliero Universitario Sant'Andrea", 41.9924, 12.4700),
    #("Presidio Ospedaliero San Filippo Neri",41.9546, 12.4157),
    ("Tiberia Hospital", 41.9533, 12.5591),
    ("Aurelia Hospital", 41.8887, 12.3976),

    # === ROMA EST ===
    ("Policlinico Tor Vergata", 41.860618822062264, 12.631796354157585), 
    ("Policlinico Casilino", 41.8758, 12.5758),


    # === ROMA SUD - LITORALE ===
    ("Campus Bio-Medico", 41.7767, 12.4694),
    ("Ospedale Grassi - Ostia", 41.7328, 12.2778),
    ("Ospedale Bambino Gesù - Palidoro", 41.9408, 12.1089),
    ("Poliambulatorio Acilia", 41.7553, 12.3547),
    #("Presidio Sanitario Fiumicino", 41.7711, 12.2356),
    
    # === CASTELLI ROMANI ===
    ("Ospedale San Sebastiano Frascati", 41.8089, 12.6806),
    #("Ospedale Civile di Velletri", 41.6878, 12.7778),
    #("Ospedale di Albano Laziale", 41.7289, 12.6583),
    #("Ospedale di Genzano", 41.7075, 12.6894),
    ("Ospedale di Marino", 41.7703, 12.6614),
    #("Ospedale di Rocca Priora", 41.7917, 12.7583),
    ("Poliambulatorio Ciampino", 41.8003, 12.6044),
    ("Ospedale di Lanuvio", 41.6756, 12.6967),
    
    # === PROVINCIA NORD ===
    #("Ospedale di Monterotondo", 42.0533, 12.6189),
    #("Ospedale Padre Pio Bracciano", 42.1006, 12.1764),
    #("Ospedale di Campagnano", 42.1331, 12.3817),
    #("Poliambulatorio Morlupo", 42.1442, 12.5042),
    #("Ospedale di Mentana", 42.0331, 12.6419),
    #("Presidio Sanitario Fonte Nuova", 41.9967, 12.6206),
    
    # === PROVINCIA EST - TIBURTINA ===
    #("Ospedale San Giovanni Evangelista Tivoli", 41.9631, 12.7978),
    ("Ospedale Coniugi Bernardini Palestrina", 41.8394, 12.8914),
    #("Ospedale di Guidonia", 41.9994, 12.7236),
    ("Ospedale di Subiaco", 41.9253, 13.0931),
    ("Presidio Sanitario Marcellina", 42.0228, 12.8056),
    ("Ospedale di San Polo dei Cavalieri", 42.0108, 12.8372),
    ("Poliambulatorio Gallicano", 41.8719, 12.9931),
    ("Presidio Sanitario Castel Madama", 41.9756, 12.8644),
    ("Ospedale di Vicovaro", 42.0144, 12.8958),
    
    # === PROVINCIA SUD - PRENESTINA/CASILINA ===
    ("Ospedale Parodi Delfino Colleferro", 41.7272, 13.0036),
    ("Ospedale di Valmontone", 41.7747, 12.9194),
    ("Presidio Sanitario Zagarolo", 41.8389, 12.8306),
    #("Ospedale di Genazzano", 41.8317, 12.9731),
    ("Poliambulatorio San Cesareo", 41.8194, 12.8028),
    ("Presidio Sanitario Labico", 41.7883, 12.8856),
    ("Ospedale di Artena", 41.7403, 12.9133),
    ("Presidio Sanitario Cave", 41.8178, 12.9347),
    
    # === PROVINCIA OVEST - AURELIA ===
    ("Ospedale San Paolo Civitavecchia", 42.0931, 11.7967),
    ("Ospedale di Cerveteri", 41.9975, 12.0958),
    ("Presidio Sanitario Ladispoli", 41.9536, 12.0733),
    ("Ospedale di Santa Marinella", 42.0347, 11.8544),
    ("Poliambulatorio Manziana", 42.1292, 12.1289),
]

PALERMO_HOSPITALS = [
    # === PALERMO CITTÀ ===
    ("ARNAS Civico Palermo", 38.1157, 13.3615),
    ("Ospedale Policlinico Paolo Giaccone", 38.1096, 13.3513),
    ("Ospedale Villa Sofia", 38.1472, 13.3237),
    ("Ospedale Cervello", 38.1508, 13.3086),
    ("Ospedale Buccheri La Ferla", 38.0936, 13.3744),
    ("ISMETT", 38.1294, 13.3378),
    ("Ospedale dei Bambini Di Cristina", 38.1119, 13.3647),

    # === PROVINCIA ===
    ("Ospedale Cimino Termini Imerese", 37.9844, 13.6953),
    ("Ospedale San Raffaele Giglio Cefalù", 38.0397, 14.0221),
    ("Ospedale Civico Partinico", 38.0481, 13.1156),
    ("Ospedale di Corleone", 37.8167, 13.3017),
    ("Ospedale di Petralia Sottana", 37.8058, 14.0889),
]


MILANO_HOSPITALS = [
    # === MILANO CITTÀ ===
    ("IRCCS Ospedale Galeazzi - Sant'Ambrogio", 45.5275, 9.0971),
    #("Ospedale Caduti Bollatesi", 45.5444, 9.1153),
    ("Ospedale di Garbagnate Milanese 'Guido Salvini' - ASST Rhodense",45.5874, 9.0940),
    #("Ospedale Sacco", 45.5220, 9.1238),
    ("ASST Grande Ospedale Metropolitano Niguarda", 45.5143, 9.1879),
    ("Ospedale Maggiore Policlinico di Milano", 45.4642, 9.1947),
    ("IRCCS Ospedale San Raffaele", 45.5129, 9.2644),
    #("Ospedale Humanitas Rozzano", 45.3819, 9.1586),
    ("Ospedale Fatebenefratelli", 45.4725, 9.1844),
    ("Ospedale San Carlo Borromeo", 45.4707, 9.1145),
    #("Ospedale Buzzi", 45.4914, 9.2111),
    ("Ospedale San Paolo", 45.4379, 9.1592),
    #("Ospedale Macedonio Melloni", 45.4683, 9.2179),
    ("Ospedale dei Bambini Vittore Buzzi", 45.4902, 9.1662),
    ("ASST Nord Milano - Ospedale Città di Sesto San Giovanni",45.5408, 9.2263),
    ("Humanitas San Pio X Ospedale e Poliambulatorio", 45.4987, 9.1896),
    ("IRCCS Istituto Clinico Humanitas Research Hospital", 45.3780, 9.1662),
    ("IRCCS Policlinico San Donato", 45.4124, 9.2772),
    ("Ospedale San Gerardo Monza", 45.6071, 9.2587),


    # === HINTERLAND ===
    ("Ospedale di Rho", 45.5306, 9.0367),
    ("Ospedale di Garbagnate", 45.587, 9.0940),
    #("Ospedale di Melegnano", 45.3575, 9.3250),
    ("Ospedale di Cernusco sul Naviglio", 45.5250, 9.3322),
    ("Ospedale di Abbiategrasso", 45.3989, 8.9197),
    ("Ospedale di Legnano", 45.5942, 8.9156),
    #("Ospedale di Sesto San Giovanni", 45.5369, 9.2383),
    #("Ospedale di Busto Arsizio", 45.6117, 8.8508),
    ("Ospedale di Magenta", 45.4647, 8.8839),
    ("Ospedale di Treviglio", 45.5158, 9.5944),
    ("Ospedale di Desio", 45.6222, 9.2106),
]


NAPOLI_HOSPITALS = [
    ("Ospedale S. Maria di Loreto",40.8483, 14.2725),
    ("Ospedale del Mare", 40.8522, 14.3447),
    ("Presidio Ospedaliero San Giovanni Bosco", 40.8748, 14.2764),
    ("ICS Hermitage Maugeri Napoli",40.8814, 14.2478),
    ("Azienda Ospedaliera Universitaria Federico II - Policlinico", 40.8693, 14.2210),
    ("Ospedale Santobono - A.O.R.N. Santobono-Pausilipon",40.8499, 14.2315),
    ("Clinica Mediterranea - Ospedale e Centro Diagnostico", 40.8292, 14.2197),
    ("Ospedale Fatebenefratelli", 40.8177, 14.1990),
    #("Clinica Sanatrix", 40.8424, 14.2152),
    ("Ospedale San Paolo Asl Napoli 1", 40.8312, 14.1832),
    ("Azienda Ospedaliera Universitaria Luigi Vanvitelli", 40.8537, 14.2513),
    ("Ospedale Vincenzo Monaldi - AOS dei Colli", 40.8705, 14.2101),
    #("Ospedale Maresca", 40.8046, 14.3847),
    #("Presidio Ospedaliero Cavalier Raffaele Apicella", 40.8535, 14.3752),
    ("Presidio Ospedaliero San Giuseppe Moscati", 40.9608, 14.2097),
    ("Ospedale Civile San Giovanni di Dio", 40.9519, 14.2729),
    ("Ospedale di Marcianise", 41.0319, 14.3143),
    ("Ospedale Maddaloni",41.0442, 14.3800),
    ("Presidio Sanitario Napoli Est Barra",40.8467, 14.3192),
    ("Presidio Ospedaliero Santa Maria delle Grazie", 40.8556, 14.0772),
    ("Ospedale San Giuliano", 40.9239, 14.2021),
    ("Presidio Ospedaliero Cavalier Raffaele Apicella",40.8588, 14.3817),
    ("Presidio Sciuti ASL Napoli 1 Centro",40.8999, 14.2380),
    ("Ospedale Marina USA", 40.9939, 14.2564),
    ("Azienda Ospedaliera Sant'Anna e San Sebastiano", 41.0920, 14.3336)
    #("Ospedale Santa Maria della Pietà", 40.9296, 14.5438)


]


EARTH_RADIUS_KM = 6371

CITIES = {
    'roma':     (ROME_HOSPITALS,    (41.9028, 12.4964)),
    'palermo':  (PALERMO_HOSPITALS, (38.035,  13.420)),
    'milano':   (MILANO_HOSPITALS,  (45.477,  9.170)),
    'napoli':   (NAPOLI_HOSPITALS,  (40.9576, 14.3202)),
}

class Node:
    def __init__(self, id, name, lat, lon, node_type):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.node_type = node_type


class Network:
    """Rete droni per area città"""

    def __init__(self, city='milano', area_size_km=40, d_max=15, num_hubs=3,
                 grid_offset_lat=0.0, grid_offset_lon=0.0):
        """
        Args:
            area_size_km: lato dell'area quadrata in km
            d_max: range massimo drone in km
            num_hubs: numero di hub da posizionare
            grid_offset_lat: offset griglia in km (positivo = nord, negativo = sud)
            grid_offset_lon: offset griglia in km (positivo = est, negativo = ovest)
        """
        self.city = city
        hospitals_data, self.center = CITIES[city]
        self._hospitals_data = hospitals_data
        self.area_size = area_size_km
        self.max_drone_range = d_max
        self.num_hubs = num_hubs
        self.grid_offset_lat = grid_offset_lat
        self.grid_offset_lon = grid_offset_lon
        
        self.nodes = {}
        self.hospitals = []
        self.stations = []
        self.hubs = []
        
        self.network_bounds = self._calculate_bounds()
        self._add_hospitals()
        self._add_stations()
        self._add_hubs()
    
    def _calculate_bounds(self):
        """Area quadrata centrata su Roma"""
        center_lat, center_lon = self.center
        
        km_to_deg_lat = 1 / 111
        km_to_deg_lon = 1 / (111 * math.cos(math.radians(center_lat)))
        
        half_size_lat = (self.area_size / 2) * km_to_deg_lat
        half_size_lon = (self.area_size / 2) * km_to_deg_lon
        
        return {
            'min_lat': center_lat - half_size_lat,
            'max_lat': center_lat + half_size_lat,
            'min_lon': center_lon - half_size_lon,
            'max_lon': center_lon + half_size_lon
        }
    
    def _point_in_bounds(self, lat, lon):
        b = self.network_bounds
        return b['min_lat'] <= lat <= b['max_lat'] and b['min_lon'] <= lon <= b['max_lon']
    
    def _add_hospitals(self):
        """Aggiunge ospedali reali di Roma"""
        for i, (name, lat, lon) in enumerate(self._hospitals_data):
            if self._point_in_bounds(lat, lon):
                node = Node(f'H_{i+1}', name, lat, lon, 'hospital')
                self.nodes[node.id] = node
                self.hospitals.append(node)
    
    def _add_stations(self):
        """Griglia di stazioni con offset configurabile"""
        cell_size_km = self.max_drone_range * 0.8
        
        # Conversione km -> gradi (diversa per lat e lon)
        # 1° lat ≈ 111 km (costante)
        # 1° lon ≈ 111 * cos(lat) km (dipende dalla latitudine)
        center_lat = self.center[0]
        cell_deg_lat = cell_size_km / 111
        cell_deg_lon = cell_size_km / (111 * math.cos(math.radians(center_lat)))
        
        b = self.network_bounds
        
        # Applica offset in gradi
        offset_lat_deg = self.grid_offset_lat / 111
        offset_lon_deg = self.grid_offset_lon / (111 * math.cos(math.radians(center_lat)))
        
        # Punto di partenza con offset
        lat_start = b['min_lat'] + offset_lat_deg
        lon_start = b['min_lon'] + offset_lon_deg
        
        rows = int(math.ceil((b['max_lat'] - b['min_lat']) / cell_deg_lat)) + 1
        cols = int(math.ceil((b['max_lon'] - b['min_lon']) / cell_deg_lon)) + 1
        
        station_id = 1
        for i in range(rows):
            for j in range(cols):
                lat = lat_start + i * cell_deg_lat
                lon = lon_start + j * cell_deg_lon
                if self._point_in_bounds(lat, lon):
                    node = Node(f'F_{station_id}', f'Facility {station_id}', lat, lon, 'facility')
                    self.nodes[node.id] = node
                    self.stations.append(node)
                    station_id += 1
    
    def _add_hubs(self):
        """Posiziona hub con KMeans basato su ospedali"""
        if not self.hospitals:
            return
        
        points = [[h.lat, h.lon] for h in self.hospitals]
        n_clusters = min(self.num_hubs, len(points))
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(points)
        
        for i, (lat, lon) in enumerate(kmeans.cluster_centers_):
            if self._point_in_bounds(lat, lon):
                node = Node(f'HUB_{i+1}', f'Hub {i+1}', lat, lon, 'hub')
                self.nodes[node.id] = node
                self.hubs.append(node)
    
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Distanza haversine tra due punti usando sklearn (ritorna km)"""
        # sklearn haversine vuole [lat, lon] in radianti
        p1 = np.radians([[lat1, lon1]])
        p2 = np.radians([[lat2, lon2]])
        return haversine_distances(p1, p2)[0, 0] * EARTH_RADIUS_KM
    
    def save_csv(self, filename):
        """Esporta nodi in CSV"""
        data = [{
            'id': node.id,
            'name': node.name,
            'lat': node.lat,
            'lon': node.lon,
            'type': node.node_type
        } for node in self.nodes.values()]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Esportati {len(data)} nodi in {filename}")

    def make_df(self):
        """DataFrame dei nodi"""
        data = [{
            'id': node.id,
            'name': node.name,
            'lat': node.lat,
            'lon': node.lon,
            'type': node.node_type
        } for node in self.nodes.values()]
        return pd.DataFrame(data)

    def visualize(self):
        """Mappa interattiva"""
        b = self.network_bounds
        center = [self.center[0], self.center[1]]
        
        m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")
        
        styles = {
            'hospital': {'color': '#E63946', 'radius': 7, 'weight': 2},
            'hub':      {'color': '#2A9D8F', 'radius': 10, 'weight': 2},
            'facility':  {'color': '#457B9D', 'radius': 4, 'weight': 1}
        }
        
        layers = {
            'hospital': folium.FeatureGroup(name='Ospedali'),
            'hub':      folium.FeatureGroup(name='Hub'),
            'facility':  folium.FeatureGroup(name='Facility')
        }
        
        # Perimetro area
        perimeter = [
            [b['min_lat'], b['min_lon']], [b['min_lat'], b['max_lon']],
            [b['max_lat'], b['max_lon']], [b['max_lat'], b['min_lon']]
        ]
        #folium.Polygon(
        #    perimeter, color='#264653', weight=2, 
        #    fill=True, fill_opacity=0.02, dash_array='8,4'
        #).add_to(m)
        
        # Nodi
        for node in self.nodes.values():
            s = styles[node.node_type]
            folium.CircleMarker(
                [node.lat, node.lon], 
                radius=s['radius'], 
                color=s['color'],
                fill=True, 
                fill_color=s['color'], 
                fill_opacity=0.7,
                weight=s['weight'],
                tooltip=f"<b>{node.name}</b>",
                popup=f"<b>{node.name}</b><br>ID: {node.id}<br>({node.lat:.4f}, {node.lon:.4f})"
            ).add_to(layers[node.node_type])
        
        for layer in layers.values():
            layer.add_to(m)
        
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Legenda + stats
        legend = f"""
        <div style="position:fixed;bottom:30px;left:30px;background:white;padding:14px 18px;
                    border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.15);font-family:system-ui;font-size:12px;">
            <div style="font-weight:600;margin-bottom:8px;font-size:13px;">Roma {self.area_size}×{self.area_size} km</div>
            <div><span style="color:#E63946">●</span> Ospedali: {len(self.hospitals)}</div>
            <div><span style="color:#2A9D8F">●</span> Hub: {len(self.hubs)}</div>
            <div><span style="color:#457B9D">●</span> Stazioni: {len(self.stations)}</div>
            <div style="margin-top:6px;color:#666;font-size:11px;">Range drone: {self.max_drone_range} km</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend))
        
        return m
    
    def summary(self):
        print(f"═══════════════════════════════════")
        print(f"  RETE DRONI ROMA {self.area_size}×{self.area_size} km")
        print(f"═══════════════════════════════════")
        print(f"  🏥 Ospedali:  {len(self.hospitals)}")
        print(f"  📦 Hub:       {len(self.hubs)}")
        print(f"  🔋 Stazioni:  {len(self.stations)}")
        print(f"  📏 Range:     {self.max_drone_range * 0.7:.1f} km")
        
        if len(self.stations) >= 2:
            s0 = self.stations[0]
            
            # Trova stazione adiacente orizzontale (stessa lat, lon diversa)
            horiz = [s for s in self.stations[1:] if abs(s.lat - s0.lat) < 0.001]
            if horiz:
                dist_h = self.haversine(s0.lat, s0.lon, horiz[0].lat, horiz[0].lon)
                print(f"  ↔️  Dist. orizzontale: {dist_h:.2f} km")
            
            # Trova stazione adiacente verticale (stessa lon, lat diversa)
            vert = [s for s in self.stations[1:] if abs(s.lon - s0.lon) < 0.001]
            if vert:
                dist_v = self.haversine(s0.lat, s0.lon, vert[0].lat, vert[0].lon)
                print(f"  ↕️  Dist. verticale:   {dist_v:.2f} km")
                
        if self.grid_offset_lat != 0 or self.grid_offset_lon != 0:
            print(f"  ↗️  Offset: {self.grid_offset_lat:+.1f}km N, {self.grid_offset_lon:+.1f}km E")
        print(f"═══════════════════════════════════")
        
    def get_distance_matrix(self):
        """Matrice distanze tra tutti i nodi usando sklearn haversine"""
        nodes_list = list(self.nodes.values())
        
        # Prepara coordinate in radianti [lat, lon]
        coords_rad = np.radians([[n.lat, n.lon] for n in nodes_list])
        
        # Calcola matrice distanze (in km)
        dist_matrix = haversine_distances(coords_rad) * EARTH_RADIUS_KM
        
        return dist_matrix, nodes_list


if __name__ == "__main__":

    network = Network(
        city='milano',
        area_size_km=50,
        d_max=7,
        num_hubs=3,
        grid_offset_lat=0,
        grid_offset_lon=0,
    )

    network.summary()
    network.visualize().save(f"{network.city}.html")
    network.save_csv(f"data/{network.city}/{network.city}.csv")
    
    # Test matrice distanze
    dist, nodes = network.get_distance_matrix()
    print(f"\nMatrice distanze: {dist.shape}")
    print(f"Min: {dist[dist > 0].min():.2f} km, Max: {dist.max():.2f} km")

    #nodes_df = network.make_df()
    #density_df = pd.read_csv('data/population_density.csv')