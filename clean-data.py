def edges_to_csv():
    f = open(file='edges.txt', mode='r')
    clean_f = open(file='edges.csv', mode='w')
    for line in f.readlines():
        split_line = line.split(' ')
        for i in range(2):
            split_line[i] = f'{split_line[i]};'
        clean_line = ''.join(split_line)
        clean_f.write(clean_line)
    f.close()
    clean_f.close()


def vertices_to_csv():
    f = open(file='stations.txt', mode='r')
    clean_f = open(file='stations.csv', mode='w')
    for line in f.readlines():
        split_line = line.split(' ')
        split_line[0] = f'{split_line[0]};'
        for i in range(1, len(split_line) - 1):
            split_line[i] = f'{split_line[i]} '
        clean_line = ''.join(split_line)
        clean_f.write(clean_line)
    f.close()
    clean_f.close()


if __name__ == '__main__':
    edges_to_csv()
    vertices_to_csv()
