user_list = { "God" , "Adam" , "Hitler" }
bad_boy_list = { "Hitler" }

print( "##### user #####" )
for user in user_list :
    print( user )
pass

#friend_list = user_list.difference( bad_boy_list )
friend_list = user_list - bad_boy_list 

print( "#### friend #####" )
for friend in friend_list :
    print( friend )
pass