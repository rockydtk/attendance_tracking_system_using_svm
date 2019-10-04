import pymysql

# Establish connection to MySQL Database

class Database:
    def __init__(self):
        host = "127.0.0.1"
        user = "root"
        password = "Pa$$w0rd"
        db = "face_schema"
        self.con = pymysql.connect(host=host, user=user, password=password, db=db, cursorclass=pymysql.cursors.DictCursor)
        self.cur = self.con.cursor()

    # Function to call record in Student Database
    def list_attended_students(self):
        self.cur.execute("SELECT * FROM attended_students")
        result = self.cur.fetchall()
        return result
    
    def delete_attended_students(self):
        self.cur.execute("DELETE FROM attended_students")
        self.con.commit()

    
    def add_new_attended_students(self, student_name, class_name, attend):
        self.cur.execute("INSERT INTO attended_students (student_name,class_name,attend) VALUES (%s, %s, %s)", student_name, class_name, attend)
        self.con.commit()

db_conn = Database()