names = [ "홍길동", "이순신", "유관순" ]

for name in names : 
    print( name )
pass

print( "###############" )
for i in range( 0, len( names ) ) : 
    print( "[%s] : %s" % ( i, names[i] ) )
pass

print( "###############" )
r = range( 0, len( names ) )
for i in r : 
    print( "[%s] : %s" % ( i, names[i] ) )
pass

print( "###############" )
i = 0 
for name in names : 
    print( "[%s] : %s" % ( i, name ) )
    i = i + 1
    #i += 1
pass