user_list = { "7711" : "홍길동" , "7722" : "이순신" , "7733" : "유관순"}
user_list = { }
user_list[ "7711" ] = "홍길동"
user_list[ "7722" ] = "이순신"
user_list[ "7733" ] = "유관순"

print( user_list )

for key in user_list :
    print( "key = %s" % key )
    user = user_list[ key ]
    print( "user = %s" % user )
pass