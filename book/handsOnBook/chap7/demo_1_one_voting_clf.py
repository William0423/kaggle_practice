# coding:utf-8


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score

def test_voting_hard(X_train, X_test, y_train, y_test):
	print "==============start hard============="
	log_clf = LogisticRegression()
	rnd_clf = RandomForestClassifier()
	svm_clf = SVC()
	voting_clf = VotingClassifier(
			estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
			voting='hard' # voting='hard'是什么意思
		)
	voting_clf.fit(X_train, y_train)
	for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
		clf.fit(X_train, y_train)
		y_pred=clf.predict(X_test)
		print (clf.__class__.__name__, accuracy_score(y_test, y_pred))


def test_voting_soft(X_train, X_test, y_train, y_test):
	print "===============start soft=================="
	log_clf = LogisticRegression(random_state=42)
	rnd_clf = RandomForestClassifier(random_state=42)
	svm_clf = SVC(probability=True, random_state=42) # 上面的方法和这个方法的主要差别在这里
	voting_clf = VotingClassifier(
	    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
	    voting='soft')
	voting_clf.fit(X_train, y_train)

	for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
	    clf.fit(X_train, y_train)
	    y_pred = clf.predict(X_test)
	    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

if __name__ == '__main__':

	X, y = make_moons(n_samples=500, noise=0.3, random_state=42) # random_state为随机数种子
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
	# print len(X_train), len(y_train), len(X_test), len(y_test)



	#############  关于硬投票和软投票：http://blog.csdn.net/yanyanyufei96/article/details/71195063  ###################
	test_voting_hard(X_train, X_test, y_train, y_test)
	test_voting_soft(X_train, X_test, y_train, y_test)
	# print X_train, y_train, X_test, y_test 
