Attendance Tracking System using Face Recognition
===
<h2>1. Introduction:</h2>
<ul>
    <li><b>What is Attendance Tracking System:</b> With the conventional attendance checking which is teacher need to call out student's name and the student answer to his/her call. It requires time and effort from both sides which sometimes cause frustration. If the attendance is taken in an exam, it surely a vital steps before even begin the exam. By using machine learning in face recognition, the technology will help to remove the frustration.</li>
    <li><b>Why use Attendance Tracking System:</b> With the conventional attendance checking which is teacher need to call out student's name and the student answer to his/her call. It requires time and effort from both sides which sometimes cause frustration. If the attendance is taken in an exam, it surely a vital steps before even begin the exam. By using machine learning in face recognition, the technology will help to remove the frustration.</li>
    
</ul>
<h2>2. How does it work?</h2>
        <ul>
                <li>Within a define time frames which associate with certain classes, a camera ,which connects to the training model, will check the video or image and then pass through the model to recognize students inside that video.</li>
        </ul>

<h2>3. Modules:</h2>
<h3>- <u>Essential:</u> Face Detection and Recognization:</h3>
<ul>
    <i>- Face Detection:</i>
    <li>Define bounding line</li>
    <li>Extract face video to images</li>
    <br>
    <i>- Face Recognize:</i>
    <li>openCV (cv2/4) to streamline webcam</li>
    <li>Use **model: Facenet / face_recognition**</li>
</ul>
<h3>
- <u>Extra Modules:</u>
</h3>
<ul>
    <li>Login: to authenticate the user rights to use the website </li>
    <li>Search: look for classes and student ID</li>
    <li>Classes and Students Management: CRUD for classes and students</li>
    <li>User Management: admin, teacher and Teacher Assisstant</li>
        <ul>
            <li>Admin: classes, students, teacher, TA management</li>
            <li>Teacher: add classes, add student to class</li>
            <li>Teacher Assisstant: view attendance</li>
            <li>Student: no access</li>        
        </ul>
    <li>Export Report:</li>
        <ul>     
            <li>Export: excel, csv base on classes or date</li>
            <li>CloudSQL: export report to chart</li>
        </ul>
</ul>
<h2>4. Time Table:</h2>
<img src="https://i.imgur.com/BQ8NtoZ.png">


<h2>5. Flowchart:</h2>

<img src="https://i.imgur.com/TxpTwAr.png">

<h2 style="text-align:center;">Add Student / course function flow</h2>
<img src="https://i.imgur.com/T81YQiF.png">

<h2>6. Reference:</h2>
<ul>
<li>Omar Faruqe and 
Md. Al Mehedi Hasan, Face recognition using PCA and SVM, (September 2009) <a href ="https://https://www.researchgate.net/publication/224599198_Face_recognition_using_PCA_and_SVM">https://https://www.researchgate.net/publication/224599198_Face_recognition_using_PCA_and_SVM</a></li>
<li>Jamim and ageitgey, Face Recognition library, (last updated 25 Aug 2019) <a href ="https://github.com/ageitgey/face_recognition">https://github.com/ageitgey/face_recognition</a></li>
<li>Dr. J.B. Helonde and Mr. Somil Mayur Shah, AUTOMATIC ATTENDANCE SYSTEM BY FACE RECOGNITION USING MACHINE LEARNING, (last updated 25 Aug 2019) <a href ="http://www.ijesrt.com/issues%20pdf%20file/Archive-2018/October-2018/17.pdf">http://www.ijesrt.com/issues%20pdf%20file/Archive-2018/October-2018/17.pdf</a></li>
</ul>
