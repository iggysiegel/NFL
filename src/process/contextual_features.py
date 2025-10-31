"""A collection of helper functions to generate additional contextual features for each
matchup."""

import numpy as np

STADIUM_COORDS = {
    "3Com Park": [37.71776358705399, -122.38265146327004],
    "AT&T Stadium": [32.7476536192796, -97.09346480674803],
    "Acrisure Stadium": [40.44665862959527, -80.01562082946674],
    "Alamo Dome": [29.416917956282518, -98.47876066077102],
    "Allegiant Stadium": [36.09058897775333, -115.1831559289777],
    "Allianz Arena": [48.21916719143793, 11.624517134826833],
    "Alltel Stadium": [30.323761699055446, -81.63773594299442],
    "Arena Corinthians": [-23.545313378722323, -46.474036086162144],
    "Arrowhead Stadium": [39.04893967527478, -94.4842179522998],
    "Azteca Stadium": [19.302830304745573, -99.15064572255206],
    "Bank of America Stadium": [35.225723701859955, -80.85310972554073],
    "Candlestick Park": [37.719800408544884, -122.38824945445245],
    "CenturyLink Field": [47.59509388983651, -122.33178960754695],
    "Cleveland Browns Stadium": [41.50598115907837, -81.6999021558453],
    "Cowboys Stadium": [32.74777098396895, -97.09294443600648],
    "Deutsche Bank Park": [50.068787266566765, 8.645075029412201],
    "Dolphin Stadium": [25.957872623603848, -80.23841776331764],
    "Edward Jones Dome": [38.63279802875273, -90.1889681443845],
    "Empower Field at Mile High": [39.74384292485632, -105.02009768903329],
    "EverBank Field": [30.32373391610112, -81.6372209588762],
    "FedExField": [38.90798209508523, -76.86443901553181],
    "FirstEnergy Stadium": [40.36518260651342, -75.93345135135631],
    "Ford Field": [42.34002223174641, -83.04579612325382],
    "GEHA Field at Arrowhead Stadium": [39.04906361590624, -94.48433142272498],
    "Georgia Dome": [33.75964405226499, -84.40348919280979],
    "Giants Stadium": [40.81348201118589, -74.07471449636785],
    "Gillette Stadium": [42.091133652832646, -71.26457171162821],
    "Hard Rock Stadium": [25.958162017326156, -80.23927607018138],
    "Heinz Field": [40.44665046484062, -80.01565301597412],
    "Hubert H. Humphrey Metrodome": [44.97339059233523, -93.25726196662487],
    "Invesco Field at Mile High": [39.74371093248997, -105.01997967183952],
    "Jacksonville Municipal Stadium": [30.32385136497278, -81.63750076627852],
    "LP Field": [36.16635637336871, -86.77123513157721],
    "Lambeau Field": [44.50123344046089, -88.0623692366009],
    "Levi's Stadium": [37.403086356906456, -121.96962416774282],
    "Lincoln Financial Field": [39.90078511084875, -75.1673202546358],
    "Los Angeles Memorial Coliseum": [34.014207155119614, -118.28820028513033],
    "Louisiana Superdome": [29.950847166107227, -90.08174846020653],
    "Lucas Oil Stadium": [39.759952152505036, -86.16427396276279],
    "Lumen Field": [47.59510112499403, -122.33088838534003],
    "M&T Bank Stadium": [39.27775030102381, -76.62217122822813],
    "Mall of America Field": [44.97339059233523, -93.25726196662487],
    "McAfee Coliseum": [37.75187225334873, -122.20064238360375],
    "Mercedes-Benz Stadium": [33.75547481552644, -84.40110548884566],
    "Mercedes-Benz Superdome": [33.75547481552644, -84.40110548884566],
    "MetLife Stadium": [40.8137012515222, -74.07495053075537],
    "Monster Park": [47.59509388983651, -122.33178960754695],
    "NRG Stadium": [29.684759158284198, -95.41091125281719],
    "New Era Field": [42.773864826617356, -78.78733708459781],
    "New Meadowlands Stadium": [40.8137012515222, -74.0749290730838],
    "Nissan Stadium": [36.166330388972455, -86.77141752178576],
    "O.co Coliseum": [37.75181287257686, -122.20056728175315],
    "Oakland-Alameda County Coliseum": [37.75181287257686, -122.20056728175315],
    "Paul Brown Stadium": [39.095832268060704, -84.5164439425055],
    "Paycor Stadium": [39.095832268060704, -84.5164439425055],
    "Pro Player Stadium": [25.957814744773998, -80.23879327257053],
    "Qualcomm Stadium": [32.78100061400963, -117.11977452382408],
    "Qwest Field": [47.5955062922181, -122.33177887871116],
    "RCA Dome": [39.75206638384812, -86.13132754149592],
    "Ralph Wilson Stadium": [42.7738884527876, -78.78730489809041],
    "Raymond James Stadium": [27.975859600808946, -82.50351679522635],
    "Reliant Stadium": [29.685085386405554, -95.41124384672688],
    "Ring Central Coliseum": [37.75138872282047, -122.20017031482868],
    "Rogers Centre": [43.64153695021876, -79.3898144422112],
    "SoFi Stadium": [33.95376125585767, -118.33964577719284],
    "Soldier Field": [41.86244900620779, -87.61709610000113],
    "Sports Authority Field at Mile High": [39.744106908830496, -105.02036590992822],
    "State Farm Stadium": [33.52788404213075, -112.26287044097782],
    "StubHub Center": [33.86481420242769, -118.26136791027315],
    "Sun Devil Stadium": [33.426291797214986, -111.93250893173051],
    "Sun Life Stadium": [25.95804625992264, -80.2392224260024],
    "TCF Bank Stadium": [44.9765022023242, -93.22494317095563],
    "TIAA Bank Stadium": [30.323817264940487, -81.63748917977111],
    "Texas Stadium": [32.84900861328897, -96.90798272336242],
    "The Coliseum": [34.0141626895074, -118.28827538698091],
    "Tiger Stadium (LSU)": [30.411988711489233, -91.18370901654338],
    "Tottenham Stadium": [51.60432020735256, -0.06656235512797312],
    "Twickenham Stadium": [51.45611621883938, -0.3418913416467443],
    "U.S. Bank Stadium": [44.973403192954784, -93.25734430033019],
    "University of Phoenix Stadium": [33.527955592030516, -112.26290262748519],
    "Wembley Stadium": [51.55599798912324, -0.2799932128032906],
}


def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate the Haversine distance between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    r = 3958.8
    return c * r


def travel_distance(data):
    """Calculate travel distance."""
    data = data.copy()
    temp = {team: [37.77, -89.54] for team in data["team"].unique()}
    home_travel_distance = []
    away_travel_distance = []

    for row in data.itertuples():
        home_team = row.team
        away_team = row.opponent_team
        try:
            lat = STADIUM_COORDS[row.stadium][0]
            lon = STADIUM_COORDS[row.stadium][1]
        except KeyError as exc:
            raise ValueError(
                f"Unable to find location for stadium: {row.stadium}."
            ) from exc
        home_travel_distance.append(
            haversine(temp[home_team][0], temp[home_team][1], lat, lon)
        )
        away_travel_distance.append(
            haversine(temp[away_team][0], temp[away_team][1], lat, lon)
        )
        temp[home_team] = [lat, lon]
        temp[away_team] = [lat, lon]

    data["home_travel_distance"] = home_travel_distance
    data["away_travel_distance"] = away_travel_distance
    data["travel_adv"] = data["away_travel_distance"] - data["home_travel_distance"]
    return data
