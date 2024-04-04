import csv
import pandas as pd

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


class Statistic:
    def __init__(self, match, player_id):
        self.match = match
        # serve and return
        self.player_id = player_id
        self.points_served = 0
        self.points_return = 0
        self.first_serve = 0
        self.first_serve_win = 0
        self.second_serve_win = 0
        self.first_return = 0
        self.first_return_win = 0
        self.second_return_win = 0
        self.serve_speeds = []
        self.avg_serve_speeds = 0

        self.first_serve_rate = 0
        self.first_serve_win_rate = 0
        self.second_serve_win_rate = 0
        self.first_return_win_rate = 0
        self.second_return_win_rate = 0

        # special points
        self.ace = 0
        self.double_fault = 0
        self.unforced_error = 0
        self.winner_shot = 0
        self.break_point = 0
        self.break_point_converted = 0
        self.break_point_faced = 0
        self.break_point_saved = 0
        self.net_point = 0
        self.net_point_win = 0

        self.ace_rate = 0
        self.double_fault_rate = 0
        self.unforced_error_rate = 0
        self.winner_shot_rate = 0
        self.break_point_converted_rate = 0
        self.break_point_saved_rate = 0
        self.net_point_win_rate = 0

        # overall
        self.total_point = 0
        self.win_point = 0
        self.lose_point = 0
        self.win_set = 0
        self.lose_set = 0
        self.win_game = 0
        self.lose_game = 0
        self.dominance = 0

        self.win_point_rate = 0

    def calc(self):
        for data_point in self.match.data_points:
            if data_point['PointNumber'] == '0' or data_point['PointServer'] == '0':
                continue
            if data_point['PointServer'] == self.player_id:
                self.points_served += 1
                if data_point['Speed_KMH'] != '0':
                    try:
                       self.serve_speeds.append(int(data_point['Speed_KMH']))
                    except:
                        pass
                if data_point['P'+self.player_id+'FirstSrvIn'] != '0' and data_point['P'+self.player_id+'FirstSrvIn'] != '1':
                    try:
                        if data_point['P'+self.player_id+'DoubleFault'] == '1':
                            self.double_fault += 1
                        elif data_point['ServeNumber'] == '1':
                            self.first_serve += 1
                            if data_point['PointWinner'] == self.player_id:
                                self.first_serve_win += 1
                        else:
                            if data_point['PointWinner'] == self.player_id:
                                self.second_serve_win += 1
                    except:
                        if data_point['P'+self.player_id+'DoubleFault'] == '1':
                            self.double_fault += 1
                        else:
                            self.first_serve += 0.5
                            self.first_serve_win += 0.30
                            self.second_serve_win += 0.20
                else:
                    if data_point['P'+self.player_id+'FirstSrvIn'] == '1':
                        self.first_serve += 1
                        if data_point['PointWinner'] == self.player_id:
                            self.first_serve_win += 1
                    else:
                        if data_point['PointWinner'] == self.player_id:
                            self.second_serve_win += 1
                    if data_point['P'+self.player_id+'DoubleFault'] == '1':
                        self.double_fault += 1
                
            else:
                self.points_return += 1
                if data_point['P'+str(3-int(self.player_id))+'DoubleFault'] == '1':
                    self.points_return -= 1
                if data_point['P'+self.player_id+'FirstSrvIn'] != '0' and data_point['P'+self.player_id+'FirstSrvIn'] != '1':
                    try:
                        if data_point['ServeNumber'] == '1':
                            self.first_return += 1
                            if data_point['PointWinner'] == self.player_id:
                                self.first_return_win += 1
                        else:
                            if data_point['PointWinner'] == self.player_id:
                                self.second_return_win += 1
                    except:
                        self.first_return += 0.5
                        self.first_return_win += 0.15
                        self.second_return_win += 0.30
                else:
                    if data_point['P'+self.player_id+'FirstSrvIn'] == '1':
                        self.first_return += 1
                        if data_point['PointWinner'] == self.player_id:
                            self.first_return_win += 1
                    else:
                        if data_point['PointWinner'] == self.player_id:
                            self.second_return_win += 1

            if data_point['P'+self.player_id+'Ace'] == '1':
                self.ace += 1
            if data_point['P'+self.player_id+'UnfErr'] == '1':
                self.unforced_error += 1
            if data_point['P'+self.player_id+'Winner'] == '1':
                self.winner_shot += 1
            if data_point['P'+self.player_id+'BreakPoint'] == '1':
                self.break_point += 1
                if data_point['PointWinner'] == self.player_id:
                    self.break_point_converted += 1
            if data_point['P'+str(3-int(self.player_id))+'BreakPoint'] == '1':
                self.break_point_faced += 1
                if data_point['PointWinner'] == self.player_id:
                    self.break_point_saved += 1
            if data_point['P'+self.player_id+'NetPoint'] == '1':
                self.net_point += 1
                if data_point['PointWinner'] == self.player_id:
                    self.net_point_win += 1
            if data_point['PointWinner'] == self.player_id:
                self.win_point += 1
            else:
                self.lose_point += 1
            self.total_point += 1
            if data_point['SetWinner'] == self.player_id:
                self.win_set += 1
            else:
                self.lose_set += 1
            if data_point['GameWinner'] == self.player_id:
                self.win_game += 1
            else:
                self.lose_game += 1

        self.avg_serve_speeds = sum(self.serve_speeds) / len(self.serve_speeds) if len(self.serve_speeds) > 0 else 0

        self.first_serve_rate = self.first_serve / self.points_served if self.points_served > 0 else 0
        self.first_serve_win_rate = self.first_serve_win / self.first_serve if self.first_serve > 0 else 0
        self.second_serve_win_rate = self.second_serve_win / (self.points_served - self.first_serve) if self.points_served - self.first_serve > 0 else 0
        self.first_return_win_rate = self.first_return_win / self.first_return if self.first_return > 0 else 0
        self.second_return_win_rate = self.second_return_win / (self.points_return - self.first_return) if self.points_return - self.first_return > 0 else 0

        self.ace_rate = self.ace / self.points_served if self.points_served > 0 else 0
        self.double_fault_rate = self.double_fault / self.points_served if self.points_served > 0 else 0
        self.unforced_error_rate = self.unforced_error / self.total_point if self.total_point > 0 else 0
        self.winner_shot_rate = self.winner_shot / self.total_point if self.total_point > 0 else 0
        self.break_point_converted_rate = self.break_point_converted / self.break_point if self.break_point > 0 else 0
        self.break_point_saved_rate = self.break_point_saved / self.break_point_faced if self.break_point_faced > 0 else 0
        self.net_point_win_rate = self.net_point_win / self.net_point if self.net_point > 0 else 0

        self.win_point_rate = self.win_point / self.total_point if self.total_point > 0 else 0
        self.dominance = (self.points_served - self.first_serve_win - self.second_serve_win) / (self.first_return_win + self.second_return_win) if (self.first_return_win + self.second_return_win) > 0 else 5.0

    def print_stats(self):
        print(f'Player {self.player_id} stats:')
        print(f'Points served: {self.points_served}, Points return: {self.points_return}')
        print(f'First serve: {self.first_serve}')
        print(f'First serve win: {self.first_serve_win}, First serve win rate: {self.first_serve_win_rate}')
        print(f'Second serve win: {self.second_serve_win}, Second serve win rate: {self.second_serve_win_rate}')
        print(f'First return win: {self.first_return_win}, First return win rate: {self.first_return_win_rate}')
        print(f'Second return win: {self.second_return_win}, Second return win rate: {self.second_return_win_rate}')
        print(f'Average serve speed: {self.avg_serve_speeds}')
        print(f'Ace: {self.ace}, Ace rate: {self.ace_rate}')
        print(f'Double fault: {self.double_fault}, Double fault rate: {self.double_fault_rate}')
        print(f'Unforced error: {self.unforced_error}, Unforced error rate: {self.unforced_error_rate}')
        print(f'Winner shot: {self.winner_shot}, Winner shot rate: {self.winner_shot_rate}')
        print(f'Break point: {self.break_point}, Break point converted: {self.break_point_converted}, Break point converted rate: {self.break_point_converted_rate}')
        print(f'Break point faced: {self.break_point_faced}, Break point saved: {self.break_point_saved}, Break point saved rate: {self.break_point_saved_rate}')
        print(f'Net point: {self.net_point}, Net point win: {self.net_point_win}, Net point win rate: {self.net_point_win_rate}')
        print(f'Total point: {self.total_point}, Win point: {self.win_point}, Lose point: {self.lose_point}, Win point rate: {self.win_point_rate}')

class Match:
    def __init__(self, match_id, overall, server_id):
        self.match_id = match_id
        self.overall = overall  # 存储比赛的整体信息
        self.data_points = []  # 存储单个比赛的数据点
        self.server_id = server_id
        self.server_stats = None
        self.returner_stats = None
        self.input_vector = []
        self.result_vector = []

    def add_data_point(self, data_point):
        self.data_points.append(data_point)

    def print_overall(self):
        print(self.match_id)
        print(f"{self.overall['player1']}({getEloRating('1',self)}) v.s. {self.overall['player2']}({getEloRating('2',self)})")

    def encoder(self):
        self.input_vector = [
            self.server_stats.first_serve_rate,
            self.server_stats.first_serve_win_rate,
            self.server_stats.second_serve_win_rate,
            self.server_stats.ace_rate,
            self.server_stats.double_fault_rate,
            self.server_stats.winner_shot_rate,
            self.server_stats.break_point_faced,
            self.server_stats.break_point_saved_rate,
            self.server_stats.net_point,
            self.server_stats.net_point_win_rate,
            self.server_stats.unforced_error_rate,
            self.server_stats.avg_serve_speeds,
            self.returner_stats.unforced_error_rate,
            self.returner_stats.first_return_win_rate,
            self.returner_stats.second_return_win_rate,
            self.returner_stats.net_point,
            self.returner_stats.net_point_win_rate,
            surface[self.match_id.split('-')[1]],
        ]
        self.result_vector = [getEloRating(self.server_id,self), getEloRating(str(3-int(self.server_id)),self)]

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
                if (match_id + '-1') not in self.matches:
                    for i in range(1,3):
                        self.matches[match_id + '-' + str(i)] = Match(match_id, self.get_overall(match_id), str(i))
                    if match_id == '2023-wimbledon-1101':
                        self.matches[match_id] = Match(match_id, self.get_overall(match_id), '0')

                if row['PointNumber'] == '0' or row['PointServer'] == '0':
                    continue
                if row['PointServer'] == '1':
                    self.matches[match_id + '-1'].add_data_point(row)
                else:
                    self.matches[match_id + '-2'].add_data_point(row)
                if match_id == '2023-wimbledon-1101':
                    self.matches[match_id].add_data_point(row)

    def get_match(self, match_id):
        return self.matches.get(match_id, None)
    
    def get_overall(self, match_id):
        match_id = match_id.split('-')[0]+'-'+match_id.split('-')[1]+'-'+match_id.split('-')[2]

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

    for match_id, _match in match_manager.matches.items():
        if match_id == '2023-wimbledon-1101':
            continue
        if match_id.split('-')[3] == '1':
            _match.server_stats = Statistic(_match, '1')
            _match.server_stats.calc()
            _match.returner_stats = Statistic(_match, '2')
            _match.returner_stats.calc()
        else:
            _match.server_stats = Statistic(_match, '2')
            _match.server_stats.calc()
            _match.returner_stats = Statistic(_match, '1')
            _match.returner_stats.calc()
    
    # match = match_manager.get_match('2023-wimbledon-1101-1')
    # match.server_stats.print_stats()
    # print('---------------------------------')
    # match.returner_stats.print_stats()
    
    # for match_id, _match in match_manager.matches.items():
    #     _match.encoder()
 
    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.metrics import mean_squared_error
    # import numpy as np

    # input_vectors = np.array([match.input_vector for match_id, match in match_manager.matches.items()])
    # result_vectors = np.array([match.result_vector for match_id, match in match_manager.matches.items()])
    # X_train, X_test, y_train, y_test = train_test_split(input_vectors, result_vectors, test_size=0.1, random_state=42)

    # # 初始化随机森林回归器
    # regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # # 训练模型
    # regressor.fit(X_train, y_train)

    # # 进行预测
    # y_pred = regressor.predict(X_test)

    # # 评估模型
    # mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    # print(f"Mean Squared Error for Player 1: {mse[0]}")
    # print(f"Mean Squared Error for Player 2: {mse[1]}")

    from joblib import dump, load
    from sklearn.ensemble import RandomForestRegressor
    # 保存模型到文件
    # dump(regressor, 'random_forest_model_split.joblib')

    loaded_model = load('random_forest_model_split.joblib')

    match_id = '2023-wimbledon-1101'
    match_manager.set_paths('tennis_slam_pointbypoint-master/2023-wimbledon-points.csv', 'tennis_slam_pointbypoint-master/2023-wimbledon-matches.csv')
    _match = Match(match_id, match_manager.get_overall(match_id),'1')

    special_match_1 = match_manager.get_match('2023-wimbledon-1101-1')
    special_match_2 = match_manager.get_match('2023-wimbledon-1101-2')

    avg_elo_pred = []
    player1_serve_elo_pred = []
    player2_return_elo_pred = []

    for row in special_match_1.data_points:
        _match.add_data_point(row)
        if len(_match.data_points) >= 6:###########################################################333
            del _match.data_points[0]
        _match.server_stats = Statistic(_match, '1')
        _match.server_stats.calc()
        _match.returner_stats = Statistic(_match, '2')
        _match.returner_stats.calc()
        _match.encoder()
        prediction = loaded_model.predict([_match.input_vector])
        # avg_elo_pred.append(prediction[0][0] + 1500)
        player1_serve_elo_pred.append(prediction[0][0])
        player2_return_elo_pred.append(prediction[0][1])

    # for i in range(len(player1_elo_pred) - 1):
    #     player1_elo_point_pred.append((i+1) * player1_elo_pred[i+1] - (i) * player1_elo_pred[i])
    #     player2_elo_point_pred.append((i+1) * player2_elo_pred[i+1] - (i) * player2_elo_pred[i])

    # import pandas
    # import matplotlib.pyplot as plt
    # plt.plot(player1_serve_elo_pred, label= special_match_1.overall['player'+special_match_1.server_id])
    # plt.plot(player2_return_elo_pred, label= special_match_1.overall['player'+str(3-int(special_match_1.server_id))])
    # # plt.plot(avg_elo_pred, label='Average')
    # plt.legend()
    # plt.show()

    # player1_elo_pred = pandas.DataFrame(player1_elo_pred)
    # player2_elo_pred = pandas.DataFrame(player2_elo_pred)
    # df=pd.concat([player1_elo_pred,player2_elo_pred],axis=1)
    # df.to_excel('elo_split_1.xlsx')

    avg_elo_pred = []
    player1_return_elo_pred = []
    player2_serve_elo_pred = []

    for row in special_match_2.data_points:
        _match.add_data_point(row)
        if len(_match.data_points) >= 6:###########################################################333
            del _match.data_points[0]
        _match.server_stats = Statistic(_match, '2')
        _match.server_stats.calc()
        _match.returner_stats = Statistic(_match, '1')
        _match.returner_stats.calc()
        _match.encoder()
        prediction = loaded_model.predict([_match.input_vector])
        # avg_elo_pred.append(prediction[0][0] + 1500)
        player1_return_elo_pred.append(prediction[0][1])
        player2_serve_elo_pred.append(prediction[0][0])
    
    # import pandas
    # import matplotlib.pyplot as plt
    # plt.plot(player1_return_elo_pred, label= special_match_2.overall['player'+special_match_2.server_id])
    # plt.plot(player2_serve_elo_pred, label= special_match_2.overall['player'+str(3-int(special_match_2.server_id))])
    # # plt.plot(avg_elo_pred, label='Average')
    # plt.legend()
    # plt.show()

    # player1_elo_pred = pandas.DataFrame(player1_elo_pred)
    # player2_elo_pred = pandas.DataFrame(player2_elo_pred)
    # df=pd.concat([player1_elo_pred,player2_elo_pred],axis=1)
    # df.to_excel('elo_split_2.xlsx')
    
    avg_elo_pred = []
    player1_elo_pred = []
    player2_elo_pred = []

    _special_match = match_manager.get_match('2023-wimbledon-1101')
    for row in _special_match.data_points:
        if row['PointNumber'] == '0' or row['PointServer'] == '0':
            continue
        if row['PointServer'] == '1':
            player1_elo_pred.append(player1_serve_elo_pred[0])
            player2_elo_pred.append(player2_return_elo_pred[0])
            del player1_serve_elo_pred[0]
            del player2_return_elo_pred[0]
        else:
            player1_elo_pred.append(player1_return_elo_pred[0])
            player2_elo_pred.append(player2_serve_elo_pred[0])
            del player1_return_elo_pred[0]
            del player2_serve_elo_pred[0]
    
    for i in range(len(player1_elo_pred) - 1):
        avg_elo_pred.append((player1_elo_pred[i] + player2_elo_pred[i]) / 2)

    import pandas
    import matplotlib.pyplot as plt
    plt.plot(player1_elo_pred, label= _special_match.overall['player1'])
    plt.plot(player2_elo_pred, label= _special_match.overall['player2'])
    plt.plot(avg_elo_pred, label='Average')
    # plt.plot(avg_elo_pred, label='Average')
    plt.legend()
    plt.show()

    player1_elo_pred = pandas.DataFrame(player1_elo_pred)
    player2_elo_pred = pandas.DataFrame(player2_elo_pred)
    df=pd.concat([player1_elo_pred,player2_elo_pred],axis=1)
    df.to_excel('elo_2.xlsx')