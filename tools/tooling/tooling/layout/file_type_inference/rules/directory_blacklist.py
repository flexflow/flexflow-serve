from tooling.layout.file_type_inference.rules.rule import Rule, IsDir, And, HasAttribute, ParentSatisfies, IsNamed, AncestorSatisfies, Not
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'directory_blacklist.deps',
        And.from_iter([
            ParentSatisfies(HasAttribute(FileAttribute.IS_PROJECT_ROOT)),
            IsDir(),
            IsNamed('deps'),
        ]),
        FileAttribute.IS_DEPS_DIR,
    ),
    Rule('directory_blacklist.lib',
         And.from_iter([
             ParentSatisfies(HasAttribute(FileAttribute.IS_PROJECT_ROOT)),
             IsDir(),
             IsNamed('lib'),
         ]),
         FileAttribute.IS_LIB_DIR,
    ),
    Rule('directory_blacklist.is_child_of_lib',
         And.from_iter([
             ParentSatisfies(HasAttribute(FileAttribute.IS_LIB_DIR)),
             IsDir(),
         ]),
         FileAttribute.IS_CHILD_OF_LIB
    ),
    Rule(
        'directory_blacklist.whitelist.find',
        AncestorSatisfies(
            And.from_iter([
                HasAttribute(FileAttribute.IS_CHILD_OF_LIB),
                IsNamed('utils'),
            ]),
        ),
        FileAttribute.IS_WHITELISTED
    ),
    Rule(
        'directory_blacklist.find',
        Not(HasAttribute(FileAttribute.IS_WHITELISTED)),
        FileAttribute.IS_BLACKLISTED,
    )
})


