import sqlite3
 
def getAddSets(values, key):
    values.add(key)
    value_at_index_is_zero = []
    index = 0
    for c in key:
        if c== '0':
            value_at_index_is_zero.append(index)
            index+=1
 
    for i in value_at_index_is_zero:
        temp = key[:i]
        temp2 = key[i+1:]
        for x in range(1,7):
            values.add(temp+str(x)+temp2)
    return values
 
 
 
#################################################
#filename = "training_data_4.db"
#database = sqlite3.connect('Documents\\DB\\' + filename)
 
database = sqlite3.connect('training_data_4.db')
db = database.cursor()
db.execute('DROP TABLE IF EXISTS viable')
db.execute('''CREATE TABLE viable (id text PRIMARY KEY)''')
trainingsset = set()
counter = 0
for name in db.execute('SELECT id from success WHERE succ == 1'):
    counter = counter+1
    trainingsset = getAddSets(trainingsset, name[0])
 
print(len(trainingsset))
print(counter,'were run successfuly')
for key in trainingsset:
    db.execute('INSERT INTO viable VALUES(?)',[key])
 
database.commit()
 
db.execute('DROP TABLE IF EXISTS training')
db.execute('''CREATE TABLE training AS SELECT tasksets.*, success.succ, success.fail FROM viable INNER JOIN success ON viable.id = success.id INNER JOIN tasksets ON viable.id = tasksets.id''')
 
database.commit()
 
 
 
"""Training Data """
 
 
 
 
"""Parameters"""
 
#id TEXT PRIMARY KEY, succ INTEGER, fail INTEGER)
 
 
filename = "training_data_4.db"
 
train = sqlite3.connect(filename)
#label = sqlite3.connect('Documents\\DB\\' + filename)
 
train_cursor = train.cursor()
#label_cursor = label.cursor()
 
print("Opened database file")
 
train_cursor.execute('SELECT * from training')
#label_cursor.execute('SELECT * from success')
#num_tasksets = db.fetchone()
train_data = train_cursor.fetchall()
print len(train_data)
print train_data[27000] 
#label_data = label_cursor.fetchall()
