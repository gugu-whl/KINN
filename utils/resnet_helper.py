import json
import requests


def request_points(sourceLayerNum, sourceXs, sourceYs, targetLayerNum, onlyOutline):
    response = requests.get("http://localhost:8090/search-point/searchIntersection", params={
        "sourceLayerNum": sourceLayerNum,
        "sourceXs": sourceXs,
        "sourceYs": sourceYs,
        "targetLayerNum": targetLayerNum,
        "onlyOutline": onlyOutline
    })
    json_result = json.loads(response.text)
    return json_result


def points_2_str(points):
    # 点坐标拼接成字符串返回
    source_xs = ''
    source_ys = ''

    point_strs = []

    for temp_pos_idx, point in enumerate(points):
        if 'x' not in point:
            print('!!!!!!!!!!')
        point_strs.append(str(point['x']) + ',' + str(point['y']))
    point_strs = set(point_strs)

    for temp_pos_idx, point_str in enumerate(point_strs):
        if temp_pos_idx == 0:
            source_xs = point_str.split(',')[0]
            source_ys = point_str.split(',')[1]
        else:
            source_xs = source_xs + ',' + point_str.split(',')[0]
            source_ys = source_ys + ',' + point_str.split(',')[1]

    return source_xs, source_ys

'''
    从矩阵中找出最大的坐标
        datas:      矩阵详细（包含多个通道）
        points:     查找范围坐标（针对所有通道）
'''
def find_max_location(datas, points):
    # 每一个通道中依次查找到的最大点坐标（去重）
    result_point_arr = []
    if len(points) == 0:
        print("!!!!!")
    for channel in datas:
        max_in_channel = None
        temp_position = None
        for point in points:
            # print(str(point['x']) + "," + str(point['y']) + ':' + str(channel[point['x']][point['y']]))
            temp_value = channel[point['x']][point['y']]
            if (max_in_channel is None) or (temp_value > max_in_channel):
                max_in_channel = temp_value
                temp_position = (point['x'], point['y'])
        result_point_arr.append(temp_position)
    result_point_arr = list(set(result_point_arr))

    # 最大点坐标拼接成字符串返回，用于下一层的推算
    source_xs = ''
    source_ys = ''
    for temp_pos_idx, temp_pos_arr in enumerate(result_point_arr):
        print('====>' + str(temp_pos_arr))
        if temp_pos_idx == 0:
            source_xs = str(temp_pos_arr[0])
            source_ys = str(temp_pos_arr[1])
        else:
            source_xs = source_xs + ',' + str(temp_pos_arr[0])
            source_ys = source_ys + ',' + str(temp_pos_arr[1])
    return source_xs, source_ys


def request_points_max(sourceLayerNum, sourceXs, sourceYs, targetLayerNum, onlyOutline, maxTargetLayerNum = 6, datas = []):

    sourceLayerNumInt = int(sourceLayerNum)
    targetLayerNumInt = int(targetLayerNum)

    temp_source_xs = sourceXs
    temp_source_ys = sourceYs

    for idx in range(sourceLayerNumInt, maxTargetLayerNum, -1):
        tempSourceLayerNum = idx
        tempTargetLayerNum = idx - 1

        temp_response = requests.get("http://localhost:8090/search-point/searchIntersection", params={
            "sourceLayerNum": tempSourceLayerNum,
            "sourceXs": temp_source_xs,
            "sourceYs": temp_source_ys,
            "targetLayerNum": tempTargetLayerNum,
            "onlyOutline": 'false'
        })
        temp_json_result = json.loads(temp_response.text)
        temp_source_xs, temp_source_ys = find_max_location(datas, tempTargetLayerNum, temp_json_result)
        # print(idx)

    response = requests.get("http://localhost:8090/search-point/searchIntersection", params={
        "sourceLayerNum": maxTargetLayerNum,
        "sourceXs": temp_source_xs,
        "sourceYs": temp_source_ys,
        "targetLayerNum": targetLayerNum,
        "onlyOutline": onlyOutline
    })
    json_result = json.loads(response.text)
    return json_result


'''
根据神经网络每层数据的结果详情逐层处理
1、逐层调用java-helper查询上一层的点数据
2、每个通道的点向上一层查询关联区域，然后根据上一层的每个通道数据详情查找这个区域最大的点（这个点作为再下一次的查询条件）
3、最后一次结果不再取最大，直接返回
'''
def request_points_by_layer_detail(sourceLayerNum, sourceXs, sourceYs, targetLayerNum, onlyOutline, layer_detail_arr, pic_idx):
    sourceLayerNumInt = int(sourceLayerNum)
    targetLayerNumInt = int(targetLayerNum)

    temp_source_xs = sourceXs
    temp_source_ys = sourceYs
    json_result = None
    for idx in range(sourceLayerNumInt, targetLayerNumInt, -1):

        tempSourceLayerNum = idx
        tempTargetLayerNum = idx - 1

        # 前面的推导方式：查找上一层的结果只取每一通道的最大值的点
        if tempTargetLayerNum > 15:
            temp_response = requests.post("http://localhost:8090/search-point/searchUnion", data={
                "sourceLayerNum": tempSourceLayerNum,
                "sourceXs": temp_source_xs,
                "sourceYs": temp_source_ys,
                "targetLayerNum": tempTargetLayerNum,
                "onlyOutline": 'false'
            })
            print('temp_source_xs:' + temp_source_xs)
            print('temp_source_ys:' + temp_source_ys)
            json_result = json.loads(temp_response.text)
            temp_source_xs, temp_source_ys = find_max_location(layer_detail_arr[tempTargetLayerNum - 1][pic_idx], json_result)
            print(str(idx) + ": " + temp_source_xs + ";" + temp_source_ys)
        else:
            # 尝试为每个方块处理为边框（暂时不考虑用这种方式了）
            if tempTargetLayerNum == 2 or tempTargetLayerNum == 1 and False:
                onlyOutline = 'false'
                if tempTargetLayerNum == 1:
                    onlyOutline = 'true'
                json_result = []
                temp_source_xs_split = temp_source_xs.split(',')
                temp_source_ys_split = temp_source_ys.split(',')
                temp_source_len = len(temp_source_xs_split)
                for temp_source_idx in range(temp_source_len):
                    print("进度: " + str(temp_source_idx) + "/" + str(temp_source_len))
                    temp_response = requests.post("http://localhost:8090/search-point/searchUnion", data={
                        "sourceLayerNum": tempSourceLayerNum,
                        "sourceXs": temp_source_xs_split[temp_source_idx],
                        "sourceYs": temp_source_ys_split[temp_source_idx],
                        "targetLayerNum": tempTargetLayerNum,
                        "onlyOutline": onlyOutline
                    })
                    json_result_temp = json.loads(temp_response.text)
                    if isinstance(json_result_temp, list):
                        json_result = json_result + json_result_temp
                    else:
                        if json_result_temp is not None:
                            json_result.append(json_result_temp)
                temp_source_xs, temp_source_ys = points_2_str(json_result)
            else:              # 最后N层的推导方式，逐个点反推，然后结果的区域取交集（每个通道的结果也取交集）
                temp_response = requests.post("http://localhost:8090/search-point/searchUnion", data={
                    "sourceLayerNum": tempSourceLayerNum,
                    "sourceXs": temp_source_xs,
                    "sourceYs": temp_source_ys,
                    "targetLayerNum": tempTargetLayerNum,
                    "onlyOutline": 'false'
                })
                json_result = json.loads(temp_response.text)
                temp_source_xs, temp_source_ys = points_2_str(json_result)
            print(str(idx) + ": " + temp_source_xs + ";" + temp_source_ys)
        if idx == targetLayerNumInt:
            break
    return json_result

if __name__ == '__main__':
    result = request_points_max("69", '4', '0', "1", 'true')
    print(len(result))

