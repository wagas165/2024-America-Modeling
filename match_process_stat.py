import csv
import pandas as pd
import time

elo_data = pd.read_csv("history elo.csv")
data_columns = elo_data.columns[1:]  # 排除'Name'列
date_columns_converted = pd.to_datetime(data_columns, errors='coerce', format='%Y/%m/%d').strftime('%Y-%m-%d')
elo_data.columns = ['Name'] + list(date_columns_converted)

surface = {
    'ausopen': 0,
    'frenchopen': 1,
    'wimbledon': 2,
    'usopen': 1
}

open_types = ['ausopen', 'frenchopen', 'wimbledon', 'usopen']


def score_calc(last_point, data_point):
    score_data = {
        'player1_sets' : last_point.score_data['player1_sets'],
        'player2_sets' : last_point.score_data['player2_sets'],
        'player1_games' : last_point.score_data['player1_games'],
        'player2_games' : last_point.score_data['player2_games'],
        'player1_points' : last_point.score_data['player1_points'],
        'player2_points' : last_point.score_data['player2_points'],
    }
    # points = ['0', '15', '30', '40', 'AD']
    if last_point.info['is_new_game']:
        score_data['player1_points'] = 0
        score_data['player2_points'] = 0

    if last_point.info['is_new_set']:
        score_data['player1_games'] = 0
        score_data['player2_games'] = 0
    
    if data_point.info['is_tie_break']:
        if data_point.info['point_winner'] == '1':
            score_data['player1_points'] += 1
        else:
            score_data['player2_points'] += 1
        
        if score_data['player1_points'] >= 7 and score_data['player1_points'] == score_data['player2_points']:
            score_data['player1_points'] = 6
            score_data['player2_points'] = 6
        
        if score_data['player1_points'] >= 7 and score_data['player1_points'] - score_data['player2_points'] >= 2:
            score_data['player1_games'] += 1
            score_data['player1_sets'] += 1

        if score_data['player2_points'] >= 7 and score_data['player2_points'] - score_data['player1_points'] >= 2:
            score_data['player2_games'] += 1
            score_data['player2_sets'] += 1
    else:
        if data_point.info['point_winner'] == '1':
            score_data['player1_points'] += 1
        else:
            score_data['player2_points'] += 1
        
        if score_data['player1_points'] == 4 and score_data['player2_points'] == 4:
            score_data['player1_points'] = 3
            score_data['player2_points'] = 3
        
        if score_data['player1_points'] >= 4 and score_data['player1_points'] - score_data['player2_points'] >= 2:
            score_data['player1_games'] += 1
            if score_data['player1_games'] >= 6 and score_data['player1_games'] - score_data['player2_games'] >= 2:
                score_data['player1_sets'] += 1

        if score_data['player2_points'] >= 4 and score_data['player2_points'] - score_data['player1_points'] >= 2:
            score_data['player2_games'] += 1
            if score_data['player2_games'] >= 6 and score_data['player2_games'] - score_data['player1_games'] >= 2:
                score_data['player2_sets'] += 1


point_print = [0, 15, 30, 40, 'AD']

class PointData:
    def __init__(self, match, data_point):
        self.data_point = data_point
    
    def print_score(self):
        print(f'{self.score_data["player1_sets"]}-{self.score_data["player2_sets"]} {self.score_data["player1_games"]}-{self.score_data["player2_games"]} {point_print[self.score_data["player1_points"]]}-{point_print[self.score_data["player2_points"]]}')
        

class Match:
    def __init__(self, match_id, overall):
        self.match_id = match_id
        self.overall = overall  # 存储比赛的整体信息
        self.data_points = []  # 存储单个比赛的数据点
        self.process = []
        self.process_game = []
        self.process_set = []

        self.player1_stats = None
        self.player2_stats = None
        self.player1_elo = getEloRating('1', self)
        self.player2_elo = getEloRating('2', self)
        self.delta_elo = self.player1_elo - self.player2_elo
        self.input_vector = []
        self.result_vector = []

        self.now_data = {
            'best_of' : 3,
            'delta_elo' : self.delta_elo,
            'point_num' : 1,
            'player1_sets' : 0,
            'player2_sets' : 0,
            'player1_games' : 0,
            'player2_games' : 0,
            'player1_points' : 0,
            'player2_points' : 0,
            'score_leader' : '0',
            'server' : '',
            'is_tie_break' : False,
            'point_winner' : '',
            'isP1Ace' : False,
            'isP2Ace' : False,
            'isP1DoubleFault' : False,
            'isP2DoubleFault' : False,
            'isP1Winner' : False,
            'isP2Winner' : False,
            'isP1UnforcedError' : False,
            'isP2UnforcedError' : False,
            'isP1NetPoint' : False,
            'isP2NetPoint' : False,
        }

    def add_data_point(self, data_point):
        self.data_points.append(data_point)

    def play_one_point(self):
        is_game_end = False
        is_set_end = False
        if self.now_data['is_tie_break']:
            if self.now_data['point_winner'] == '1':
                self.now_data['player1_points'] += 1
                self.now_data['point_num'] += 1
            elif self.now_data['point_winner'] == '2':
                self.now_data['player2_points'] += 1
                self.now_data['point_num'] += 1

            if self.now_data['player1_points'] >= 7 and self.now_data['player1_points'] == self.now_data['player2_points']:
                self.now_data['player1_points'] = 6
                self.now_data['player2_points'] = 6

            if self.now_data['player1_points'] >= 7 and self.now_data['player1_points'] - self.now_data['player2_points'] >= 2:
                self.now_data['player1_games'] += 1
                self.now_data['player1_sets'] += 1
                self.now_data['is_tie_break'] = False
                self.now_data['player1_points'] = 0
                self.now_data['player2_points'] = 0
                self.now_data['player1_games'] = 0
                self.now_data['player2_games'] = 0
                is_game_end = True
                is_set_end = True

            if self.now_data['player2_points'] >= 7 and self.now_data['player2_points'] - self.now_data['player1_points'] >= 2:
                self.now_data['player2_games'] += 1
                self.now_data['player2_sets'] += 1
                self.now_data['is_tie_break'] = False
                self.now_data['player1_points'] = 0
                self.now_data['player2_points'] = 0
                self.now_data['player1_games'] = 0
                self.now_data['player2_games'] = 0
                is_game_end = True
                is_set_end = True

        else:
            if self.now_data['point_winner'] == '0':
                return
            
            if self.now_data['point_winner'] == '1':
                self.now_data['player1_points'] += 1
                self.now_data['point_num'] += 1
            else:
                self.now_data['player2_points'] += 1
                self.now_data['point_num'] += 1

            if self.now_data['player1_points'] == 4 and self.now_data['player2_points'] == 4:
                self.now_data['player1_points'] = 3
                self.now_data['player2_points'] = 3

            if self.now_data['player1_points'] >= 4 and self.now_data['player1_points'] - self.now_data['player2_points'] >= 2:
                self.now_data['player1_games'] += 1
                self.now_data['player1_points'] = 0
                self.now_data['player2_points'] = 0
                is_game_end = True
                if self.now_data['player1_games'] >= 6 and self.now_data['player1_games'] - self.now_data['player2_games'] >= 2:
                    self.now_data['player1_sets'] += 1
                    self.now_data['player1_games'] = 0
                    self.now_data['player2_games'] = 0
                    is_set_end = True
                elif self.now_data['player1_games'] == 6 and self.now_data['player2_games'] == 6:
                    self.now_data['is_tie_break'] = True
                
            if self.now_data['player2_points'] >= 4 and self.now_data['player2_points'] - self.now_data['player1_points'] >= 2:
                self.now_data['player2_games'] += 1
                self.now_data['player1_points'] = 0
                self.now_data['player2_points'] = 0
                is_game_end = True
                if self.now_data['player2_games'] >= 6 and self.now_data['player2_games'] - self.now_data['player1_games'] >= 2:
                    self.now_data['player2_sets'] += 1
                    self.now_data['player1_games'] = 0
                    self.now_data['player2_games'] = 0
                    is_set_end = True
                elif self.now_data['player1_games'] == 6 and self.now_data['player2_games'] == 6:
                    self.now_data['is_tie_break'] = True
        
        if is_game_end:
            self.now_data['game_winner'] = self.now_data['point_winner']
            while len(self.process) > 0:
                self.process[0]['game_winner'] = self.now_data['game_winner']
                self.process_game.append(self.process[0].copy())
                del self.process[0]
            # self.process_game.append(self.now_data.copy())
        
        if is_set_end:
            self.now_data['set_winner'] = self.now_data['point_winner']
            while len(self.process_game) > 0:
                self.process_game[0]['set_winner'] = self.now_data['set_winner']
                self.process_set.append(self.process_game[0].copy())
                del self.process_game[0]
            # self.process_set.append(self.now_data.copy())
    
    def print_score(self):
        print(f'{self.now_data["player1_sets"]}-{self.now_data["player2_sets"]} {self.now_data["player1_games"]}-{self.now_data["player2_games"]}',
              f'{point_print[self.now_data["player1_points"]]}-{point_print[self.now_data["player2_points"]]}, Point_num: {self.now_data["point_num"]},',
              f'game_winner: {self.now_data["game_winner"]}, set_winner: {self.now_data["set_winner"]}, leader: {self.now_data["score_leader"]}',
              f'Server: {self.now_data["server"]}, Winner:{self.now_data["point_winner"]}, IsTieBreak:{self.now_data["is_tie_break"]}',
              f'IsP1Ace:{self.now_data["isP1Ace"]}, IsP2Ace:{self.now_data["isP2Ace"]},IsP1DF:{self.now_data["isP1DoubleFault"]}, IsP2DF:{self.now_data["isP2DoubleFault"]}, IsP1W:{self.now_data["isP1Winner"]}, IsP2W:{self.now_data["isP2Winner"]}',
              f'IsP1UnfE:{self.now_data["isP1UnforcedError"]}, IsP2UnfE:{self.now_data["isP2UnforcedError"]}, IsP1NP:{self.now_data["isP1NetPoint"]}, IsP2NP:{self.now_data["isP2NetPoint"]}')
    
    def process_data(self):
        self.now_data = {
            'best_of' : 3,
            'delta_elo' : self.delta_elo,
            'point_num' : 1,
            'player1_sets' : 0,
            'player2_sets' : 0,
            'player1_games' : 0,
            'player2_games' : 0,
            'player1_points' : 0,
            'player2_points' : 0,
            'score_leader' : '0',
            'server' : '',
            'is_tie_break' : False,
            'point_winner' : '',
            'isP1Ace' : False,
            'isP2Ace' : False,
            'isP1DoubleFault' : False,
            'isP2DoubleFault' : False,
            'isP1Winner' : False,
            'isP2Winner' : False,
            'isP1UnforcedError' : False,
            'isP2UnforcedError' : False,
            'isP1NetPoint' : False,
            'isP2NetPoint' : False,
            'game_winner' : '',
            'set_winner' : '',
            'match_winner' : '',
        }
        data_point = self.data_points[0]
        self.now_data['server'] = self.data_points[0]['PointServer']
        self.now_data['point_winner'] = self.data_points[0]['PointWinner']
        if self.now_data['player1_sets'] > self.now_data['player2_sets']:
            self.now_data['score_leader'] = '1'
        elif self.now_data['player1_sets'] < self.now_data['player2_sets']:
            self.now_data['score_leader'] = '2'
        else:
            if self.now_data['player1_games'] > self.now_data['player2_games']:
                self.now_data['score_leader'] = '1'
            elif self.now_data['player1_games'] < self.now_data['player2_games']:
                self.now_data['score_leader'] = '2'
            else:
                self.now_data['score_leader'] = '0'

        if self.now_data['point_winner'] == '1' or self.now_data['point_winner'] == '2':
            self.now_data['isP1Ace'] = self.data_points[0]['P1Ace'] == '1'
            self.now_data['isP2Ace'] = self.data_points[0]['P2Ace'] == '1'
            self.now_data['isP1DoubleFault'] = self.data_points[0]['P1DoubleFault'] == '1'
            self.now_data['isP2DoubleFault'] = self.data_points[0]['P2DoubleFault'] == '1'
            self.now_data['isP1Winner'] = self.data_points[0]['P1Winner'] == '1'
            self.now_data['isP2Winner'] = self.data_points[0]['P2Winner'] == '1'
            self.now_data['isP1UnforcedError'] = self.data_points[0]['P1UnfErr'] == '1'
            self.now_data['isP2UnforcedError'] = self.data_points[0]['P2UnfErr'] == '1'
            self.now_data['isP1NetPoint'] = self.data_points[0]['P1NetPoint'] == '1'
            self.now_data['isP2NetPoint'] = self.data_points[0]['P2NetPoint'] == '1'
    
        if self.now_data['point_winner'] == '1' or self.now_data['point_winner'] == '2':
            self.process.append(self.now_data.copy())
        for data_point in self.data_points[1:]:
            self.play_one_point()
            self.now_data['server'] = data_point['PointServer']
            self.now_data['point_winner'] = data_point['PointWinner']
            if self.now_data['player1_sets'] > self.now_data['player2_sets']:
                self.now_data['score_leader'] = '1'
            elif self.now_data['player1_sets'] < self.now_data['player2_sets']:
                self.now_data['score_leader'] = '2'
            else:
                if self.now_data['player1_games'] > self.now_data['player2_games']:
                    self.now_data['score_leader'] = '1'
                elif self.now_data['player1_games'] < self.now_data['player2_games']:
                    self.now_data['score_leader'] = '2'
                else:
                    self.now_data['score_leader'] = '0'
            if self.now_data['point_winner'] == '1' or self.now_data['point_winner'] == '2':
                self.now_data['isP1Ace'] = data_point['P1Ace'] == '1'
                self.now_data['isP2Ace'] = data_point['P2Ace'] == '1'
                self.now_data['isP1DoubleFault'] = data_point['P2DoubleFault'] == '1'
                self.now_data['isP2DoubleFault'] = data_point['P2DoubleFault'] == '1'
                self.now_data['isP1Winner'] = data_point['P1Winner'] == '1'
                self.now_data['isP2Winner'] = data_point['P2Winner'] == '1'
                self.now_data['isP1UnforcedError'] = data_point['P1UnfErr'] == '1'
                self.now_data['isP2UnforcedError'] = data_point['P2UnfErr'] == '1'
                self.now_data['isP1NetPoint'] = data_point['P1NetPoint'] == '1'
                self.now_data['isP2NetPoint'] = data_point['P2NetPoint'] == '1'
            if self.now_data['point_winner'] == '1' or self.now_data['point_winner'] == '2':
                self.process.append(self.now_data.copy())
        self.play_one_point()
        for data_point in self.process_set:
            data_point['match_winner'] = self.process_set[-1]['set_winner']
            if self.process_set[-1]['player1_sets'] + self.process_set[-1]['player2_sets'] > 3:
                data_point['best_of'] = 5
        
def print_score(now_data):
        if now_data['is_tie_break'] == False:
            print(  f'{now_data["player1_sets"]}-{now_data["player2_sets"]} {now_data["player1_games"]}-{now_data["player2_games"]}',
                    f'{point_print[now_data["player1_points"]]}-{point_print[now_data["player2_points"]]}, Point_num: {now_data["point_num"]},',
                    f'game_winner: {now_data["game_winner"]}, set_winner: {now_data["set_winner"]}, match_winner: {now_data["match_winner"]}, leader: {now_data["score_leader"]}, ',
                    f'Server: {now_data["server"]}, Winner:{now_data["winner"]}, IsTieBreak:{now_data["is_tie_break"]}',
                    # f'IsP1Ace:{now_data["isP1Ace"]}, IsP2Ace:{now_data["isP2Ace"]},IsP1DF:{now_data["isP1DoubleFault"]}, IsP2DF:{now_data["isP2DoubleFault"]}, IsP1W:{now_data["isP1Winner"]}, IsP2W:{now_data["isP2Winner"]}',
                    # f'IsP1UnfE:{now_data["isP1UnforcedError"]}, IsP2UnfE:{now_data["isP2UnforcedError"]}, IsP1NP:{now_data["isP1NetPoint"]}, IsP2NP:{now_data["isP2NetPoint"]}'
                    )
        else:
            print(  f'{now_data["player1_sets"]}-{now_data["player2_sets"]} {now_data["player1_games"]}-{now_data["player2_games"]}',
                    f'{now_data["player1_points"]}-{now_data["player2_points"]}, Point_num: {now_data["point_num"]},',
                    f'game_winner: {now_data["game_winner"]}, set_winner: {now_data["set_winner"]}, match_winner: {now_data["match_winner"]}, leader: {now_data["score_leader"]}, ',
                    f'Server: {now_data["server"]}, Winner:{now_data["winner"]}, IsTieBreak:{now_data["is_tie_break"]}',
                    # f'IsP1Ace:{now_data["isP1Ace"]}, IsP2Ace:{now_data["isP2Ace"]},IsP1DF:{now_data["isP1DoubleFault"]}, IsP2DF:{now_data["isP2DoubleFault"]}, IsP1W:{now_data["isP1Winner"]}, IsP2W:{now_data["isP2Winner"]}',
                    # f'IsP1UnfE:{now_data["isP1UnforcedError"]}, IsP2UnfE:{now_data["isP2UnforcedError"]}, IsP1NP:{now_data["isP1NetPoint"]}, IsP2NP:{now_data["isP2NetPoint"]}'
                    )
        
class MatchManager:
    def __init__(self):
        self.matches = {}  # 以match_id为键，Match实例为值
        self.points_path = ''
        self.matches_path = ''

    def set_paths(self, points_path, matches_path):
        self.points_path = points_path
        self.matches_path = matches_path
    
    def load_matches(self):
        with open(self.points_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                match_id = row['match_id']
                if match_id.split('-')[2][0] == '2':
                    continue
                if match_id not in self.matches:    
                    self.matches[match_id] = Match(match_id, self.get_overall(match_id))
                self.matches[match_id].add_data_point(row)

    def get_match(self, match_id):
        return self.matches.get(match_id, None)
    
    def get_overall(self, match_id):
        with open(self.matches_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if row['match_id'] == match_id:
                    return row
        return None

def getEloRating(player_id, match):
    player_name = match.overall['player' + player_id]
    match_year = match.overall['match_id'].split('-')[0]
    match_type = match.overall['match_id'].split('-')[1]
    player_data = elo_data[elo_data['Name'] == player_name]
    if match_type == 'ausopen':
        date_range = pd.date_range(start = match_year + '-01-15', end = match_year + '-02-14').strftime('%Y-%m-%d')
    elif match_type == 'frenchopen':
        date_range = pd.date_range(start = match_year + '-05-15', end = match_year + '-06-14').strftime('%Y-%m-%d')
    elif match_type == 'wimbledon':
        date_range = pd.date_range(start = match_year + '-06-15', end = match_year + '-07-14').strftime('%Y-%m-%d')
    elif match_type == 'usopen':
        date_range = pd.date_range(start = match_year + '-08-15', end = match_year + '-09-14').strftime('%Y-%m-%d')
    
    existing_dates = [date for date in date_range if date in player_data.columns]
    player_period_data = player_data[existing_dates]
    try:
        average_elo = player_period_data.mean(axis=1).values[0]
    except:
        average_elo = 2100.0
    return round(average_elo, 1)
    
if __name__ == '__main__':
    match_manager = MatchManager()
    for year in range(2023, 2024):
        for open_type in open_types:
            try:
                points_path = f'tennis_slam_pointbypoint-master/{year}-{open_type}-points.csv'
                matches_path = f'tennis_slam_pointbypoint-master/{year}-{open_type}-matches.csv'
                match_manager.set_paths(points_path, matches_path)
                match_manager.load_matches()
                print(f'{year}-{open_type}: {len(match_manager.matches)}')
            except FileNotFoundError:
                continue
    print('---------------------------------')

    # match = match_manager.get_match('2023-wimbledon-1132')
    # match.process_data()
    
    # for data_point in match.process:
    #     print_score(data_point)

    del_list = []
    for match_id, _match in match_manager.matches.items():
        _match.process_data()
        # print(_match.match_id, end = ' ')
        if len(_match.process_set) == 0:
            del_list.append(match_id)
            pass
        # else:
        #     print_score(_match.process_set[-1])
    
    for match_id in del_list:
        del match_manager.matches[match_id]
        print(f'{match_id} deleted')

    all_data = []
    for match_id, _match in match_manager.matches.items():
        for data_point in _match.process_set:
            all_data.append(data_point)
            all_data.append({
                'best_of' : data_point['best_of'],
                'delta_elo' : -data_point['delta_elo'],
                'point_num' : data_point['point_num'],
                'player1_sets' : data_point['player2_sets'],
                'player2_sets' : data_point['player1_sets'],
                'player1_games' : data_point['player2_games'],
                'player2_games' : data_point['player1_games'],
                'player1_points' : data_point['player2_points'],
                'player2_points' : data_point['player1_points'],
                'score_leader' : '0' if data_point['score_leader'] == '0' else '1' if data_point['score_leader'] == '2' else '2',
                'server' : '0' if data_point['score_leader'] == '0' else '1' if data_point['score_leader'] == '2' else '2',
                'is_tie_break' : data_point['is_tie_break'],
                'point_winner' : '1' if data_point['point_winner'] == '2' else '2' if data_point['point_winner'] == '1' else '0',
                'isP1Ace' : data_point['isP2Ace'],
                'isP2Ace' : data_point['isP1Ace'],
                'isP1DoubleFault' : data_point['isP2DoubleFault'],
                'isP2DoubleFault' : data_point['isP1DoubleFault'],
                'isP1Winner' : data_point['isP2Winner'],
                'isP2Winner' : data_point['isP1Winner'],
                'isP1UnforcedError' : data_point['isP2UnforcedError'],
                'isP2UnforcedError' : data_point['isP1UnforcedError'],
                'isP1NetPoint' : data_point['isP2NetPoint'],
                'isP2NetPoint' : data_point['isP1NetPoint'],
                'game_winner' : '1' if data_point['game_winner'] == '2' else '2' if data_point['game_winner'] == '1' else '0',
                'set_winner' : '1' if data_point['set_winner'] == '2' else '2' if data_point['set_winner'] == '1' else '0',
                'match_winner' : '1' if data_point['match_winner'] == '2' else '2' if data_point['match_winner'] == '1' else '0'
            })
    
def prob_calc(best_of, elo_diff, is_server, is_tie_break, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets):
    if player1_points == 0 and player2_points == 0:
        return match_win_prob_calc(best_of, elo_diff, is_server, is_tie_break, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets)
    if is_tie_break:
        a = game_win_prob_calc(elo_diff, is_server, True, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets)
        b = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, 0, 0, player1_sets + 1, player2_sets)
        c = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, 0, 0, player1_sets, player2_sets + 1)
        return a * b + (1 - a) * c
    else:
        a = game_win_prob_calc(elo_diff, is_server, False, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets)
        if player1_games + 1 < 6 and player2_games + 1 < 6:
            b = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, player1_games + 1, player2_games, player1_sets, player2_sets)
            c = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, player1_games, player2_games + 1, player1_sets, player2_sets)
            return a * b + (1 - a) * c
        elif player1_games + 1 == 6 and player2_games + 1 < 6:
            b = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, 0, 0, player1_sets + 1, player2_sets)
            c = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, 5, player2_games + 1, player1_sets, player2_sets)
            return a * b + (1 - a) * c
        elif player1_games + 1 < 6 and player2_games + 1 == 6:
            b = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, player1_games + 1, 5, player1_sets, player2_sets)
            c = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, 0, 0, player1_sets, player2_sets + 1)
            return a * b + (1 - a) * c
        elif player1_games + 1 == 6 and player2_games + 1 == 6:
            b = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, 6, 5, player1_sets, player2_sets)
            c = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, 5, 6, player1_sets, player2_sets)
            return a * b + (1 - a) * c
        elif player1_games + 1 == 6 and player2_games + 1 == 7:
            b = match_win_prob_calc(best_of, elo_diff, not is_server, True, 0, 0, 6, 6, player1_sets, player2_sets)
            c = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, 0, 0, player1_sets, player2_sets + 1)
            return a * b + (1 - a) * c
        elif player1_games + 1 == 7 and player2_games + 1 == 6:
            b = match_win_prob_calc(best_of, elo_diff, not is_server, False, 0, 0, 0, 0, player1_sets + 1, player2_sets)
            c = match_win_prob_calc(best_of, elo_diff, not is_server, True, 0, 0, 6, 6, player1_sets, player2_sets)
            return a * b + (1 - a) * c




def game_win_prob_calc(elo_diff, is_server, is_tie_break, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets):
    num_A_game = 0
    num_B_game = 0

    leader = '0'
    if player1_sets > player2_sets:
        leader = '1'
    elif player1_sets < player2_sets:
        leader = '2'
    else:
        if player1_games > player2_games:
            leader = '1'
        elif player1_games < player2_games:
            leader = '2'
        else:
            leader = '0'

    if abs(elo_diff) < 50:
        for data_point in all_data:
            if abs(data_point['delta_elo']) < 50 and elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points and leader == data_point['score_leader']:
                num_B_game += 1
                if data_point['game_winner'] == '1':
                    num_A_game += 1
    elif abs(elo_diff) < 120:
        for data_point in all_data:
            if abs(data_point['delta_elo']) < 120 and elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points and leader == data_point['score_leader']:
                num_B_game += 1
                if data_point['game_winner'] == '1':
                    num_A_game += 1
    else:
        for data_point in all_data:
            if elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points and leader == data_point['score_leader']:
                num_B_game += 1
                if data_point['game_winner'] == '1':
                    num_A_game += 1
    return num_A_game / num_B_game if num_B_game > 0 else 0

def point_win_prob_calc(elo_diff, is_server, is_tie_break, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets):
    num_A_point = 0
    num_B_point = 0

    leader = '0'
    if player1_sets > player2_sets:
        leader = '1'
    elif player1_sets < player2_sets:
        leader = '2'
    else:
        if player1_games > player2_games:
            leader = '1'
        elif player1_games < player2_games:
            leader = '2'
        else:
            leader = '0'

    if abs(elo_diff) < 50:
        for data_point in all_data:
            if abs(data_point['delta_elo']) < 50 and elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points and leader == data_point['score_leader']:
                num_B_point += 1
                if data_point['point_winner'] == '1':
                    num_A_point += 1
    elif abs(elo_diff) < 120:
        for data_point in all_data:
            if abs(data_point['delta_elo']) < 120 and elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points and leader == data_point['score_leader']:
                num_B_point += 1
                if data_point['point_winner'] == '1':
                    num_A_point += 1
    else:
        for data_point in all_data:
            if elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points and leader == data_point['score_leader']:
                num_B_point += 1
                if data_point['point_winner'] == '1':
                    num_A_point += 1
            if elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points and leader == data_point['score_leader']:
                num_B_game += 1
                if data_point['game_winner'] == '1':
                    num_A_game += 1

    return num_A_point / num_B_point if num_B_point > 0 else 0

def set_win_prob_calc(elo_diff, is_server, is_tie_break, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets):
    num_A_set = 0
    num_B_set = 0

    if abs(elo_diff) < 50:
        for data_point in all_data:
            if abs(data_point['delta_elo']) < 50 and elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points:
                if data_point['player1_games'] == player1_games and data_point['player2_games'] == player2_games:
                    num_B_set += 1
                    if data_point['set_winner'] == '1':
                        num_A_set += 1
    elif abs(elo_diff) < 120:
        for data_point in all_data:
            if abs(data_point['delta_elo']) < 120 and elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points:
                if data_point['player1_games'] == player1_games and data_point['player2_games'] == player2_games:
                    num_B_set += 1
                    if data_point['set_winner'] == '1':
                        num_A_set += 1
    else:
        for data_point in all_data:
            if elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points:
                if data_point['player1_games'] == player1_games and data_point['player2_games'] == player2_games:
                    num_B_set += 1
                    if data_point['set_winner'] == '1':
                        num_A_set += 1

    return num_A_set / num_B_set if num_B_set > 0 else 0

def match_win_prob_calc(best_of, elo_diff, is_server, is_tie_break, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets):
    if player1_sets == best_of // 2 + 1:
        return 1
    if player2_sets == best_of // 2 + 1:
        return 0
    num_A_match = 0
    num_B_match = 0


    if abs(elo_diff) < 50:
        for data_point in all_data:
            if data_point['best_of'] != best_of:
                continue
            if abs(data_point['delta_elo']) < 50 and elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break:
                if data_point['player1_sets'] == player1_sets and data_point['player2_sets'] == player2_sets:
                    if data_point['player1_games'] == player1_games and data_point['player2_games'] == player2_games and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points:
                        num_B_match += 1
                        if data_point['match_winner'] == '1':
                            num_A_match += 1
    elif abs(elo_diff) < 120:
        for data_point in all_data:
            if data_point['best_of'] != best_of:
                continue
            if abs(data_point['delta_elo']) < 120 and elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break:
                if data_point['player1_sets'] == player1_sets and data_point['player2_sets'] == player2_sets:
                    if data_point['player1_games'] == player1_games and data_point['player2_games'] == player2_games and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points:
                        num_B_match += 1
                        if data_point['match_winner'] == '1':
                            num_A_match += 1
    else:
        for data_point in all_data:
            if data_point['best_of'] != best_of:
                continue
            if elo_diff * data_point['delta_elo'] >= 0 and ((data_point['server'] == '1') == is_server) and data_point['is_tie_break'] == is_tie_break:
                if data_point['player1_sets'] == player1_sets and data_point['player2_sets'] == player2_sets:
                    if data_point['player1_games'] == player1_games and data_point['player2_games'] == player2_games and data_point['player1_points'] == player1_points and data_point['player2_points'] == player2_points:
                        num_B_match += 1
                        if data_point['match_winner'] == '1':
                            num_A_match += 1

    # print (num_A_match, num_B_match)
    return num_A_match / num_B_match if num_B_match > 0 else 0

def plus_one_ball(point_winner, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets, is_tie_break, is_server):
    if is_tie_break:
        if point_winner == '1':
            player1_points += 1
            if player1_points >= 7 and player1_points - player2_points >= 2:
                is_tie_break = False
                player1_points = 0
                player2_points = 0
                player1_games = 0
                player2_games = 0
                player1_sets += 1
                is_server = not is_server
        else:
            player2_points += 1
            if player2_points >= 7 and player2_points - player1_points >= 2:
                is_tie_break = False
                player1_points = 0
                player2_points = 0
                player1_games = 0
                player2_games = 0
                player2_sets += 1
                is_server = not is_server

        if player1_points == 7 and player2_points == 7:
            player1_points = 6
            player2_points = 6
            is_server = not is_server
    else:
        if point_winner == '1':
            player1_points += 1
            if player1_points == 4 and player2_points == 4:
                player1_points = 3
                player2_points = 3
            if player1_points >= 4 and player1_points - player2_points >= 2:
                player1_games += 1
                player1_points = 0
                player2_points = 0
                is_server = not is_server
                if player1_games >= 6 and player1_games - player2_games >= 2:
                    player1_sets += 1
                    player1_games = 0
                    player2_games = 0
                elif player1_games == 6 and player2_games == 6:
                    is_tie_break = True
        else:
            player2_points += 1
            if player2_points == 4 and player1_points == 4:
                player2_points = 3
                player1_points = 3
            if player2_points >= 4 and player2_points - player1_points >= 2:
                player2_games += 1
                player1_points = 0
                player2_points = 0
                is_server = not is_server
                if player2_games >= 6 and player2_games - player1_games >= 2:
                    player2_sets += 1
                    player1_games = 0
                    player2_games = 0
                elif player1_games == 6 and player2_games == 6:
                    is_tie_break = True
    return player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets, is_tie_break, is_server

# def leverage_calc(best_of, elo_diff, is_server, is_tie_break, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets):
#     new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server = plus_one_ball('1', player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets, is_tie_break, is_server)
#     print(new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server)
#     prob1 = prob_calc(best_of, elo_diff, new_is_server, new_is_tie_break, new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets)
#     new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server = plus_one_ball('2', player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets, is_tie_break, is_server)
#     print(new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server)
#     prob2 = prob_calc(best_of, elo_diff, new_is_server, new_is_tie_break, new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets)
#     print(prob1, prob2)
#     return prob1 - prob2

# print(leverage_calc(3, 0, True, False, 0, 0, 0, 0, 0, 0))
# print('----------------')
# print(leverage_calc(3, 150, True, False, 3, 1, 5, 1, 0, 0))

# match = match_manager.get_match('2023-wimbledon-1701')
# for data_point in match.process_set:
#     print(leverage_calc(5, 0, data_point['server'] == '1', data_point['is_tie_break'], data_point['player1_points'], data_point['player2_points'], data_point['player1_games'], data_point['player2_games'], data_point['player1_sets'], data_point['player2_sets']))

# import time
# start_time = time.time()
# print(prob_calc(5, 0, True, False, 0, 0, 0, 0, 0, 0))
# print('Time: ', time.time() - start_time)
# start_time = time.time()
# print(prob_calc(3, 150, True, False, 0, 0, 0, 0, 0, 0))
# print('Time: ', time.time() - start_time)
# start_time = time.time()
# print(prob_calc(3, 0, True, False, 0, 0, 0, 0, 0, 1))
# print('Time: ', time.time() - start_time)
# start_time = time.time()
# print(prob_calc(5, 0, False, False, 0, 0, 0, 0, 2, 0))
# print('Time: ', time.time() - start_time)
# start_time = time.time()
# print(prob_calc(5, 0, True, False, 0, 0, 0, 0, 2, 0))
# print('Time: ', time.time() - start_time)
# start_time = time.time()
# print(prob_calc(5, 0, True, False, 1, 2, 3, 1, 1, 2))
# print('Time: ', time.time() - start_time)
# start_time = time.time()
# print(prob_calc(5, 150, True, True, 1, 2, 6, 6, 1, 2))
# print('Time: ', time.time() - start_time)

def serve_win_rate(match):
    player_1_serve = 0
    player_2_serve = 0
    player_1_serve_win = 0
    player_2_serve_win = 0
    for data_point in match.process_set:
        if data_point['server'] == '1':
            player_1_serve += 1
            if data_point['point_winner'] == '1':
                player_1_serve_win += 1
        elif data_point['server'] == '2':
            player_2_serve += 1
            if data_point['point_winner'] == '2':
                player_2_serve_win += 1
    return player_1_serve_win / player_1_serve, player_2_serve_win / player_2_serve

import random
def monte_carlo_simulation(serve_win_rate_1, serve_win_rate_2, best_of, is_server, is_tie_break, player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets):
    player1_win = 0
    player2_win = 0
    for i in range(10000):
        new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server = player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets, is_tie_break, is_server
        while new_player1_sets < best_of // 2 + 1 and new_player2_sets < best_of // 2 + 1:
            if new_is_server:
                str_ = random.choices(['1', '2'], [serve_win_rate_1, 1 - serve_win_rate_1])[0]
                new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server = plus_one_ball(str_, new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server)
            else:
                str_ = random.choices(['1', '2'], [1 - serve_win_rate_2, serve_win_rate_2])[0]
                new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server = plus_one_ball(str_, new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server)
            # print(str_, new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server)
        if new_player1_sets == best_of // 2 + 1:
            player1_win += 1
        elif new_player2_sets == best_of // 2 + 1:
            player2_win += 1
    return player1_win / (player1_win + player2_win), player2_win / (player1_win + player2_win)




def monte_carlo_leverage(a, b, best_of, is_server, is_tie_break, player1_points, player2_points, player1_games,
                         player2_games, player1_sets, player2_sets):
    new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server = plus_one_ball(
        '1', player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets, is_tie_break,
        is_server)
    prob1 = \
        monte_carlo_simulation(a, b, best_of, new_is_server, new_is_tie_break, new_player1_points, new_player2_points,
                               new_player1_games, new_player2_games, new_player1_sets, new_player2_sets)[0]
    new_player1_points, new_player2_points, new_player1_games, new_player2_games, new_player1_sets, new_player2_sets, new_is_tie_break, new_is_server = plus_one_ball(
        '2', player1_points, player2_points, player1_games, player2_games, player1_sets, player2_sets, is_tie_break,
        is_server)
    prob2 = \
        monte_carlo_simulation(a, b, best_of, new_is_server, new_is_tie_break, new_player1_points, new_player2_points,
                               new_player1_games, new_player2_games, new_player1_sets, new_player2_sets)[0]
    return prob1 - prob2

match_data=pd.read_csv('/Users/zhangyichi/Desktop/pythonProject/match_analsis(new)/tennis_slam_pointbypoint-master/2023-wimbledon-points.csv')
match_names=match_data['match_id'].unique()
match_names=['2023-wimbledon-1103', '2023-wimbledon-1104',
             '2023-wimbledon-1105', '2023-wimbledon-1106', '2023-wimbledon-1107', '2023-wimbledon-1108',
             '2023-wimbledon-1109', '2023-wimbledon-1110', '2023-wimbledon-1111',
             '2023-wimbledon-1113', '2023-wimbledon-1114', '2023-wimbledon-1115', '2023-wimbledon-1116',
             '2023-wimbledon-1117', '2023-wimbledon-1118', '2023-wimbledon-1119', '2023-wimbledon-1120',
             '2023-wimbledon-1121', '2023-wimbledon-1122', '2023-wimbledon-1123', '2023-wimbledon-1124',
             '2023-wimbledon-1125', '2023-wimbledon-1126', '2023-wimbledon-1127', '2023-wimbledon-1128',
             '2023-wimbledon-1129', '2023-wimbledon-1130', '2023-wimbledon-1131', '2023-wimbledon-1132',
             '2023-wimbledon-1133', '2023-wimbledon-1134', '2023-wimbledon-1135', '2023-wimbledon-1136',
             '2023-wimbledon-1137', '2023-wimbledon-1138', '2023-wimbledon-1139', '2023-wimbledon-1140',
             '2023-wimbledon-1141', '2023-wimbledon-1142', '2023-wimbledon-1143', '2023-wimbledon-1144',
             '2023-wimbledon-1145', '2023-wimbledon-1146', '2023-wimbledon-1147', '2023-wimbledon-1148',
             '2023-wimbledon-1149', '2023-wimbledon-1150', '2023-wimbledon-1151', '2023-wimbledon-1152',
             '2023-wimbledon-1153', '2023-wimbledon-1154', '2023-wimbledon-1155', '2023-wimbledon-1156',
             '2023-wimbledon-1157', '2023-wimbledon-1158', '2023-wimbledon-1159', '2023-wimbledon-1160',
             '2023-wimbledon-1161', '2023-wimbledon-1162', '2023-wimbledon-1163', '2023-wimbledon-1164',
             '2023-wimbledon-1201', '2023-wimbledon-1202', '2023-wimbledon-1203', '2023-wimbledon-1204',
             '2023-wimbledon-1205', '2023-wimbledon-1206', '2023-wimbledon-1207', '2023-wimbledon-1208',
             '2023-wimbledon-1209', '2023-wimbledon-1210', '2023-wimbledon-1211', '2023-wimbledon-1212',
             '2023-wimbledon-1213', '2023-wimbledon-1214', '2023-wimbledon-1215', '2023-wimbledon-1216',
             '2023-wimbledon-1217', '2023-wimbledon-1218', '2023-wimbledon-1219', '2023-wimbledon-1220',
             '2023-wimbledon-1221', '2023-wimbledon-1222', '2023-wimbledon-1223', '2023-wimbledon-1224',
             '2023-wimbledon-1225', '2023-wimbledon-1226', '2023-wimbledon-1227', '2023-wimbledon-1228',
             '2023-wimbledon-1229', '2023-wimbledon-1230', '2023-wimbledon-1231', '2023-wimbledon-1232',
             '2023-wimbledon-1301', '2023-wimbledon-1302', '2023-wimbledon-1303', '2023-wimbledon-1304',
             '2023-wimbledon-1305', '2023-wimbledon-1306', '2023-wimbledon-1307', '2023-wimbledon-1308',
             '2023-wimbledon-1309', '2023-wimbledon-1310', '2023-wimbledon-1311', '2023-wimbledon-1312',
             '2023-wimbledon-1313', '2023-wimbledon-1314', '2023-wimbledon-1315', '2023-wimbledon-1316',
             '2023-wimbledon-1401', '2023-wimbledon-1402', '2023-wimbledon-1403', '2023-wimbledon-1404',
             '2023-wimbledon-1405', '2023-wimbledon-1406', '2023-wimbledon-1407', '2023-wimbledon-1408',
             '2023-wimbledon-1501', '2023-wimbledon-1502', '2023-wimbledon-1503', '2023-wimbledon-1504',
             '2023-wimbledon-1601', '2023-wimbledon-1602', '2023-wimbledon-1701']

for name in match_names:
    start_time=time.time()

    match = match_manager.get_match(name)
    # print(serve_win_rate(match))
    a, b = serve_win_rate(match)
    # print(a, b)
    # print(monte_carlo_simulation(a, b, 5, True, False, 0, 0, 0, 0, 0, 0))
    # print(monte_carlo_simulation(a, b, 5, True, False, 1, 0, 0, 0, 0, 0)[0] -
    #       monte_carlo_simulation(a, b, 5, True, False, 0, 1, 0, 0, 0, 0)[0])

    ####计算levergae
    import time

    leverage = []
    i = 1
    for data_point in match.process_set:
        print(f"{i}/{len(match.process_set)}", end=' ')
        i += 1
        # start_time = time.time()
        leverage.append(monte_carlo_leverage(a, b, 5, data_point['server'] == '1', data_point['is_tie_break'],
                                             data_point['player1_points'], data_point['player2_points'],
                                             data_point['player1_games'], data_point['player2_games'],
                                             data_point['player1_sets'], data_point['player2_sets']))
        # print('Time: ', time.time() - start_time)

    # import matplotlib.pyplot as plt
    # plt.plot(leverage)
    # plt.show()

    # import csv
    # with open('leverage.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(leverage)

    # import csv
    # with open('leverage.csv', 'r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         leverage = [float(x) for x in row]

    a = 0
    b = 1
    i = 1
    alpha = 0.9
    momentum = []

    for data_point in match.process_set:
        c = 1 if data_point['point_winner'] == '1' else -1
        a = alpha * a + leverage[i - 1] * c
        b = alpha * b + 1
        momentum.append(a / b)
        i += 1

    ###将数据添加进wimberlodon中
    indice = match_data.index[match_data['match_id'] == name].tolist()[2]
    match_data.loc[range(indice,indice+len(leverage)),'leverage']=leverage
    match_data.loc[range(indice,indice+len(momentum)),'momentum']=momentum

    match_data.to_csv('/Users/zhangyichi/Desktop/pythonProject/match_analsis(new)/tennis_slam_pointbypoint-master/2023-wimbledon-points.csv',index=False)
    end_time=time.time()
    print(f'成功完成一局的统计,用时{end_time-start_time}秒')
# import matplotlib.pyplot as plt
# plt.plot(momentum)
# plt.show()
